"""The Project model — a project is a folder of files (invariant #1).

Every stage reads/writes plain files under ``projects/<id>/``; a human can hand-edit
any file and re-run from there. ``project.json`` is the single state record: status,
current stage, per-stage status, the cost log (invariant #7), and model config.

State lives in files, never only in memory or a DB. ``save()`` writes atomically so an
interrupted run never leaves a half-written ``project.json`` (invariant #3).
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

# The folder tree from PLAN.md. Created on `Project.create`.
SUBDIRS = [
    "story",
    "bible",
    "references/uploads",
    "storyboard/prompts",
    "storyboard/keyframes",
    "assets/clips",
    "assets/audio",
    "assets/qc",
    "edit",
    "output",
]

DEFAULT_ROOT = Path("projects")


class CostCapExceeded(RuntimeError):
    """Raised when a paid generation would run after the per-project cap is spent."""

    def __init__(self, total: float, cap: float | None):
        self.total = total
        self.cap = cap
        super().__init__(f"cost cap exceeded: ${total:.2f} spent of ${cap} cap")


class ProjectConfigurationConflict(ValueError):
    """An exact-idea project already exists with incompatible persisted settings."""


def slugify(text: str) -> str:
    """Lowercase, hyphenated slug; safe as a directory name."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:48] or "project"


@dataclass
class Project:
    project_id: str
    root: Path  # parent dir that holds <project_id>/
    stages: list[str]
    current_stage: str
    status: str = "in_progress"
    stage_statuses: dict[str, str] = field(default_factory=dict)
    cost_log: list[dict] = field(default_factory=list)
    cost_cap: float | None = None
    model_config: dict = field(default_factory=dict)
    idea: str = ""

    # ------------------------------------------------------------------ paths
    @property
    def dir(self) -> Path:
        return self.root / self.project_id

    @property
    def story_dir(self) -> Path:
        return self.dir / "story"

    @property
    def json_path(self) -> Path:
        return self.dir / "project.json"

    def path(self, *parts: str) -> Path:
        return self.dir.joinpath(*parts)

    # --------------------------------------------------------------- creation
    @classmethod
    def create(
        cls,
        idea: str,
        *,
        root: str | os.PathLike = DEFAULT_ROOT,
        stages: list[str],
        cost_cap: float | None = None,
        model_config: dict | None = None,
    ) -> "Project":
        """Create the project tree, or load it if it already exists (idempotent)."""
        root = Path(root)
        project_id = slugify(idea) + "-" + uuid.uuid4().hex[:6]

        # Reuse an existing project for the exact same idea (no duplicate trees).
        existing = _find_existing(root, slugify(idea), idea)
        if existing is not None:
            proj = cls.load(existing)
            conflicts = _configuration_conflicts(
                proj.model_config,
                dict(model_config or {}),
            )
            if conflicts:
                keys = ", ".join(sorted(conflicts))
                raise ProjectConfigurationConflict(
                    f"project '{proj.project_id}' already uses different configuration for: "
                    f"{keys}. Resume it or use a distinct idea for a new run."
                )
            proj._sync_with_pipeline(
                stages=stages,
                cost_cap=cost_cap,
                model_config=model_config,
            )
            return proj

        proj = cls(
            project_id=project_id,
            root=root,
            stages=list(stages),
            current_stage=stages[0],
            cost_cap=cost_cap,
            model_config=dict(model_config or {}),
            idea=idea,
        )
        for sub in SUBDIRS:
            (proj.dir / sub).mkdir(parents=True, exist_ok=True)
        proj.save()
        return proj

    # ------------------------------------------------------------- stage state
    def stage_status(self, stage: str) -> str:
        return self.stage_statuses.get(stage, "pending")

    def set_stage_status(self, stage: str, status: str) -> None:
        self.stage_statuses[stage] = status

    def _sync_with_pipeline(
        self,
        *,
        stages: list[str],
        cost_cap: float | None,
        model_config: dict | None,
    ) -> None:
        """Bring an existing project up to the currently implemented pipeline."""
        changed = False

        for stage in stages:
            if stage not in self.stages:
                self.stages.append(stage)
                changed = True

        if cost_cap is not None and self.cost_cap is None:
            self.cost_cap = cost_cap
            changed = True

        for key, value in dict(model_config or {}).items():
            if key not in self.model_config:
                self.model_config[key] = value
                changed = True

        next_stage = next(
            (stage for stage in self.stages if self.stage_status(stage) != "approved"),
            None,
        )
        if self.current_stage != next_stage:
            self.current_stage = next_stage
            changed = True

        next_status = "done" if next_stage is None else "in_progress"
        if self.status != next_status:
            self.status = next_status
            changed = True

        if changed:
            self.save()

    # -------------------------------------------------------------------- cost
    def add_cost(self, *, stage: str, provider: str, cost_usd: float, seconds: float) -> None:
        self.cost_log.append(
            {
                "stage": stage,
                "provider": provider,
                "cost_usd": cost_usd,
                "seconds": seconds,
                "ts": time.time(),
            }
        )
        self.save()

    def add_generation_cost(self, *, stage: str, generation) -> None:
        """Persist a provider generation with auditable pricing provenance."""

        tracking = dict(generation.meta.get("cost_tracking") or {})
        tracking.setdefault("usage", dict(generation.meta.get("usage") or {}))
        tracking.setdefault("estimate", True)
        if "pricing" not in tracking:
            tracking["pricing"] = {
                "model": str(generation.model),
                "status": "unpriced",
            }
            tracking.setdefault("usage_missing", not bool(tracking["usage"]))

        protected = {"stage", "provider", "cost_usd", "seconds", "ts"}
        if protected.intersection(tracking):
            raise ValueError(
                "cost tracking metadata cannot replace ledger identity fields"
            )
        entry = {
            "stage": stage,
            "provider": generation.provider,
            "cost_usd": float(generation.cost_usd),
            "seconds": float(generation.seconds),
            "ts": time.time(),
            **tracking,
        }
        json.dumps(entry)
        self.cost_log.append(entry)
        self.save()

    def total_cost(self) -> float:
        return sum(entry["cost_usd"] for entry in self.cost_log)

    def has_incomplete_cost_history(self) -> bool:
        """Return whether any ledger entry predates provenance or lacks usage."""

        return any(
            "estimate" not in entry or bool(entry.get("usage_missing"))
            for entry in self.cost_log
        )

    def within_cost_cap(self) -> bool:
        if self.cost_cap is None:
            return True
        return self.total_cost() < self.cost_cap

    def assert_budget_available(self) -> None:
        """Pre-spend guard (invariant #7): raise if the cap is already spent.

        Stages call this immediately before a paid generation so a re-run or a long
        per-shot loop stops at the cap instead of overspending. Because real costs are
        only known after a call returns, at most the generation that crosses the cap
        completes; the next call is blocked here.
        """
        if not self.within_cost_cap():
            raise CostCapExceeded(self.total_cost(), self.cost_cap)

    # --------------------------------------------------------------- (de)serial
    def to_dict(self) -> dict:
        return {
            "project_id": self.project_id,
            "idea": self.idea,
            "status": self.status,
            "stages": self.stages,
            "current_stage": self.current_stage,
            "stage_statuses": self.stage_statuses,
            "cost_log": self.cost_log,
            "cost_cap": self.cost_cap,
            "model_config": self.model_config,
        }

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.json_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2))
        tmp.replace(self.json_path)  # atomic on the same filesystem

    @classmethod
    def load(cls, project_dir: str | os.PathLike) -> "Project":
        project_dir = Path(project_dir)
        data = json.loads((project_dir / "project.json").read_text())
        project = cls(
            project_id=data["project_id"],
            root=project_dir.parent,
            stages=data["stages"],
            current_stage=data["current_stage"],
            status=data.get("status", "in_progress"),
            stage_statuses=data.get("stage_statuses", {}),
            cost_log=data.get("cost_log", []),
            cost_cap=data.get("cost_cap"),
            model_config=data.get("model_config", {}),
            idea=data.get("idea", ""),
        )
        if project._migrate_prompt_gate_stages():
            project.save()
        return project

    def _migrate_prompt_gate_stages(self) -> bool:
        """Insert prompt-review stages and infer only artifact-backed approvals.

        Migration is intentionally non-destructive: it writes state/manifests only and
        never moves, rewrites, or regenerates paid media.
        """
        changed = False
        old_stages = list(self.stages)
        old_current = self.current_stage

        migrated_stages = list(self.stages)
        owner = next(
            (stage for stage in ("storyboard", "clip") if stage in migrated_stages),
            None,
        )
        if owner is not None:
            migrated_stages = [stage for stage in migrated_stages if stage != "keyframes"]
            migrated_stages.insert(migrated_stages.index(owner) + 1, "keyframes")
        if "video" in migrated_stages:
            migrated_stages = [
                stage for stage in migrated_stages if stage != "video_prompts"
            ]
            migrated_stages.insert(migrated_stages.index("video"), "video_prompts")
        if migrated_stages != self.stages:
            self.stages = migrated_stages
            changed = True
        if not changed:
            return False

        shots_path = self.path("storyboard", "shots.json")
        shots = []
        if shots_path.is_file():
            try:
                shots = [
                    shot
                    for shot in json.loads(shots_path.read_text()).get("shots", [])
                    if isinstance(shot, dict) and shot.get("id")
                ]
            except (json.JSONDecodeError, AttributeError):
                shots = []

        planning = "storyboard" if "storyboard" in old_stages else "clip"
        planning_advanced = (
            self.stage_status(planning) == "approved"
            or (
                old_current in old_stages
                and planning in old_stages
                and old_stages.index(old_current) > old_stages.index(planning)
            )
            or old_current is None
        )
        all_keyframe_prompts = bool(shots) and all(
            self._migration_keyframe_prompt(shot).is_file() for shot in shots
        )
        all_keyframes = bool(shots) and all(
            self.path("storyboard", "keyframes", str(shot.get("keyframe") or "")).is_file()
            for shot in shots
        )
        all_video_prompts = bool(shots) and all(
            self.path("storyboard", "prompts", f"{shot['id']}.video.md").is_file()
            for shot in shots
        )
        all_clips = bool(shots) and all(
            self.path("assets", "clips", f"{shot['id']}.mp4").is_file()
            for shot in shots
        )

        if all_keyframes and all_keyframe_prompts and planning_advanced:
            if self._write_migrated_prompt_approval("keyframes"):
                self.set_stage_status("keyframes", "approved")
            else:
                self.set_stage_status(planning, "complete")
        elif all_keyframe_prompts and not all_keyframes and planning_advanced:
            self.set_stage_status(planning, "complete")

        if all_video_prompts and all_clips:
            if self._write_migrated_prompt_approval("videos"):
                self.set_stage_status("video_prompts", "approved")
            else:
                self.set_stage_status("video_prompts", "complete")
        elif all_video_prompts and not all_clips:
            self.set_stage_status("video_prompts", "complete")

        next_stage = next(
            (stage for stage in self.stages if self.stage_status(stage) != "approved"),
            None,
        )
        self.current_stage = next_stage
        self.status = "done" if next_stage is None else "in_progress"
        return True

    def _migration_keyframe_prompt(self, shot: dict) -> Path:
        suffix = "grid.md" if isinstance(shot.get("motion_grid"), dict) else "keyframe.md"
        return self.path("storyboard", "prompts", f"{shot['id']}.{suffix}")

    def _write_migrated_prompt_approval(self, kind: str) -> bool:
        from ..prompt_approvals import PromptApprovalError, confirm_prompt_batch

        try:
            confirm_prompt_batch(self, kind, confirmer="migration")
        except (PromptApprovalError, OSError, ValueError):
            return False
        return True


def _find_existing(root: Path, slug: str, idea: str) -> Path | None:
    if not root.is_dir():
        return None
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "project.json").is_file():
            if not child.name.startswith(slug + "-"):
                continue
            data = json.loads((child / "project.json").read_text())
            if data.get("idea") == idea:
                return child
    return None


def _configuration_conflicts(existing: dict, requested: dict) -> dict[str, tuple[object, object]]:
    return {
        key: (existing[key], value)
        for key, value in requested.items()
        if key in existing and existing[key] != value
    }
