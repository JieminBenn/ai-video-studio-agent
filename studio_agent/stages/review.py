"""Review stage: automated QC of every generated clip before approval (invariant #9).

Each clip is checked against story alignment, shot instructions, continuity, identity,
motion/anatomy, artifacts, audio mismatch, and safety, and a timestamped report is
written under ``assets/qc/<id>.json``. If any clip fails with a blocking issue the
stage returns ``failed`` so the orchestrator's gate blocks auto-approval and recommends
regeneration.

Idempotent (invariants #3/#5): skips the whole stage if complete; a per-clip report is
reused while its clip and story-aware QC context are unchanged (hashes are stored in
the report) and re-run only when the report is missing, the clip changed, or review
criteria changed — so a re-run never duplicates a paid check unnecessarily and a
regenerated clip is always re-reviewed.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from ..audio_native import has_audio_stream
from ..providers.base import AUDIO_QC_DIMENSIONS, qc_report_status
from .base import Providers, Stage, StageResult

QC_CONTEXT_VERSION = 5


class ReviewStage(Stage):
    name = "review"

    def run(self, project, providers: Providers) -> StageResult:
        if project.stage_status(self.name) == "complete":
            return StageResult(status="skipped", message="review already complete")

        clips = json.loads(project.path("assets", "clips", "clips.json").read_text())["clips"]
        shots = self._shots_by_id(project)

        any_failure = False
        summary = []
        for entry in clips:
            shot_id = entry["id"]
            report = self._review_clip(
                project, entry, shots.get(shot_id, {}), providers
            )
            any_failure = any_failure or not report["overall_pass"]
            summary.append({
                "id": shot_id,
                "overall_pass": report["overall_pass"],
                "recommendation": report["recommendation"],
            })

        project.path("assets", "qc", "summary.json").write_text(
            json.dumps({"clips": summary, "all_passed": not any_failure}, indent=2)
        )

        if any_failure:
            return StageResult(status="failed", message="QC found serious issues — regenerate")
        return StageResult(status="complete", message=f"QC passed: {len(clips)} clip(s)")

    def _shots_by_id(self, project) -> dict:
        shots_path = project.path("storyboard", "shots.json")
        if not shots_path.is_file():
            return {}
        return {s["id"]: s for s in json.loads(shots_path.read_text())["shots"]}

    def _review_clip(self, project, entry, shot, providers: Providers) -> dict:
        shot_id = entry["id"]
        report_path = project.path("assets", "qc", f"{shot_id}.json")
        prompt = self._prompt_for(project, shot_id, shot)
        audio_mode = str(project.model_config.get("audio_mode") or "legacy")
        native_mode = audio_mode == "native_video"
        review_metadata = self._review_metadata(
            providers, audio_mode=audio_mode
        )
        prompt_hash = self._review_context_hash(prompt, review_metadata)

        if native_mode:
            clip_path = self._confined_clip_artifact(project, entry.get("clip"))
        else:
            # Legacy projects retain their existing manifest/path behavior.
            clip_path = project.path("assets", "clips", entry["clip"])
        clip_hash = self._hash(clip_path) if clip_path is not None else ""

        audio_path = None
        native_audio_source = None
        native_audio_hash = None
        if native_mode:
            sidecar = self._confined_clip_artifact(
                project, entry.get("native_audio")
            )
            candidates = [sidecar, clip_path]
            for candidate in candidates:
                if (
                    candidate is not None
                    and candidate.is_file()
                    and has_audio_stream(candidate)
                ):
                    audio_path = candidate
                    native_audio_source = candidate.name
                    native_audio_hash = self._hash(candidate)
                    break

        # Reuse an existing report only while both the clip and story-aware QC context
        # are unchanged. Native review also keys reuse on its actual soundtrack.
        # Bump QC_CONTEXT_VERSION when review criteria change.
        if report_path.is_file():
            existing = json.loads(report_path.read_text())
            if (
                existing.get("clip_hash") == clip_hash
                and existing.get("qc_context_hash") == prompt_hash
                and existing.get("qc_context_version") == QC_CONTEXT_VERSION
                and (
                    not native_mode
                    or (
                        existing.get("native_audio_source") == native_audio_source
                        and existing.get("native_audio_hash") == native_audio_hash
                    )
                )
            ):
                return existing

        if native_mode and (clip_path is None or audio_path is None):
            report = self._missing_native_audio_report(entry["clip"], prompt)
            return self._write_report(
                report_path,
                report,
                clip_hash=clip_hash,
                prompt_hash=prompt_hash,
                review_metadata=review_metadata,
                native_audio_source=native_audio_source,
                native_audio_hash=native_audio_hash,
            )

        project.assert_budget_available()  # honor the cap before a paid QC call
        gen = providers.vlm.review(str(clip_path), prompt=prompt)
        project.add_generation_cost(stage=self.name, generation=gen)

        report = dict(gen.content)
        if native_mode:
            report = self._enforce_audio_review(
                report,
                supports_audio_review=providers.vlm.supports_audio_review,
            )
        return self._write_report(
            report_path,
            report,
            clip_hash=clip_hash,
            prompt_hash=prompt_hash,
            review_metadata=review_metadata,
            native_audio_source=native_audio_source,
            native_audio_hash=native_audio_hash,
        )

    @staticmethod
    def _review_metadata(providers: Providers, *, audio_mode: str) -> dict:
        vlm = providers.vlm
        provider_name = str(
            getattr(vlm, "name", "")
            or f"{type(vlm).__module__}.{type(vlm).__qualname__}"
        )
        return {
            "audio_mode": audio_mode,
            "vlm_provider": provider_name,
            "vlm_model": str(getattr(vlm, "model", "") or ""),
            "vlm_supports_audio_review": bool(vlm.supports_audio_review),
        }

    @staticmethod
    def _review_context_hash(prompt: str, review_metadata: dict) -> str:
        context = {"prompt": prompt, **review_metadata}
        serialized = json.dumps(
            context, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def _confined_clip_artifact(project, value: Any) -> Path | None:
        """Resolve only a direct child of assets/clips, never an escaping path."""
        raw = str(value or "")
        relative = Path(raw)
        if (
            not raw
            or relative.is_absolute()
            or len(relative.parts) != 1
            or relative.name != raw
            or raw in {".", ".."}
        ):
            return None
        clips_dir = project.path("assets", "clips")
        candidate = clips_dir / raw
        try:
            if candidate.resolve(strict=False).parent != clips_dir.resolve():
                return None
        except (OSError, RuntimeError):
            return None
        return candidate

    @staticmethod
    def _missing_native_audio_report(clip_name: str, prompt: str) -> dict:
        checks = [
            {
                "dimension": dimension,
                "passed": False,
                "severity": "high",
                "detail": "native audio stream missing or undecodable",
                "status": "failed",
                "timestamp": None,
            }
            for dimension in AUDIO_QC_DIMENSIONS
        ]
        return {
            "clip": clip_name,
            "prompt": prompt,
            "checks": checks,
            "summary": "Native audio is missing or undecodable.",
            **qc_report_status(checks),
        }

    @staticmethod
    def _enforce_audio_review(report: dict, *, supports_audio_review: bool) -> dict:
        checks = [dict(check) for check in report.get("checks", [])]
        by_dimension = {check.get("dimension"): check for check in checks}
        for dimension in AUDIO_QC_DIMENSIONS:
            if supports_audio_review and dimension in by_dimension:
                continue
            replacement = {
                "dimension": dimension,
                "passed": False,
                "severity": "high",
                "detail": (
                    "selected QC provider cannot inspect audio"
                    if not supports_audio_review
                    else "audio-capable QC response omitted this required dimension"
                ),
                "status": "not_checked",
                "timestamp": None,
            }
            existing = by_dimension.get(dimension)
            if existing is None:
                checks.append(replacement)
            else:
                checks[checks.index(existing)] = replacement
        report["checks"] = checks
        report.update(qc_report_status(checks))
        return report

    @staticmethod
    def _write_report(
        report_path,
        report,
        *,
        clip_hash: str,
        prompt_hash: str,
        review_metadata: dict,
        native_audio_source: str | None,
        native_audio_hash: str | None,
    ) -> dict:
        report["clip_hash"] = clip_hash
        report["qc_context_hash"] = prompt_hash
        report["qc_context_version"] = QC_CONTEXT_VERSION
        report.update(review_metadata)
        if review_metadata["audio_mode"] == "native_video":
            report["native_audio_source"] = native_audio_source
            report["native_audio_hash"] = native_audio_hash
        report["reviewed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2))
        return report

    def _prompt_for(self, project, shot_id, shot) -> str:
        video_prompt = self._text(project.path("storyboard", "prompts", f"{shot_id}.video.md"))
        keyframe_prompt = self._text(
            project.path("storyboard", "prompts", f"{shot_id}.keyframe.md")
        )
        idea = self._text(project.path("story", "idea.md")).strip()
        plot = self._json(project.path("story", "plot.json"))
        script = self._json(project.path("story", "script.json"))
        shots = self._shots(project)
        prev_shot, next_shot = self._neighbors(shots, shot_id)
        scene = self._script_scene(script, shot.get("scene"))

        return "\n\n".join(
            section for section in [
                f"# QC context - {shot_id}",
                "## Project intent\n"
                f"Idea: {idea or 'n/a'}\n"
                f"Language: {project.model_config.get('language', 'n/a')}\n"
                f"Format: {self._format_line(project.model_config.get('product_format'))}",
                "## Story target\n" + self._plot_summary(plot),
                "## Script scene\n" + self._json_block(scene or {}),
                "## Shot under review\n" + self._json_block(self._compact_shot(shot)),
                self._bible_identity(project, shot),
                "## Neighboring shots\n"
                f"Previous shot: {self._neighbor_line(prev_shot)}\n"
                f"Next shot: {self._neighbor_line(next_shot)}",
                "## Keyframe prompt\n" + (keyframe_prompt or "(missing)"),
                "## Video prompt\n" + (
                    video_prompt
                    or shot.get("keyframe_prompt")
                    or shot.get("description", "")
                    or "(missing)"
                ),
                "## Review priorities\n"
                "1. Does the video tell the intended story beat and match the script scene?\n"
                "2. Does it follow the exact shot instructions, action, camera, dialogue, and duration?\n"
                "3. Does it maintain continuity with neighboring shots?\n"
                "4. identity_drift: does every character match the locked Bible character "
                "identity above — same face, hair, body, wardrobe, and palette? Flag it when a "
                "character no longer matches the Bible (wrong face/hair/wardrobe, off-palette, "
                "or a 'dont' rule violated). If no Bible identity is given, judge consistency "
                "within the shot and against neighboring shots.\n"
                "5. motion_anatomy: flag anatomically or physically implausible results — a "
                "twisted or over-rotated head/neck, impossible joint angles, broken or extra/"
                "missing limbs or fingers, melted or warped faces, or a pose no real body could "
                "hold. Also flag a scene that is not logically/physically coherent (objects "
                "floating, people clipping through props, impossible scale, or nonsensical "
                "staging).\n"
                "6. Are artifacts, audio sync, and safety acceptable?\n"
                "Audio policy: verify exact speaker and dialogue, reject invented speech, "
                "confirm synchronized diegetic SFX and natural ambience, and fail "
                "music_absence if any score, song, singing, beat, melody, or underscore "
                "is audible.\n"
                "Treat minor texture shimmer as a warning; regenerate for story mismatch, wrong subject/action, "
                "continuity break, identity drift from the Bible, broken or implausible motion/anatomy, "
                "illogical staging, safety, or medium/high artifacts.",
            ] if section
        ) + "\n"

    def _bible_identity(self, project, shot) -> str:
        """Locked Bible identity for the shot's characters, so QC can verify against it.

        Threads each named character's canonical face/body/hair/wardrobe/palette and
        do/don't rules into the QC context. Without this, ``identity_drift`` has nothing
        to compare against (invariant #4)."""
        names = {
            name
            for key in ("reference_characters", "characters")
            for name in (shot.get(key) or [])
        }
        chars_dir = project.path("bible", "characters")
        if not names or not chars_dir.is_dir():
            return ""

        entries = []
        for cdir in sorted(chars_dir.iterdir()):
            cfile = cdir / "character.json"
            if not cfile.is_file():
                continue
            data = json.loads(cfile.read_text())
            name = data.get("name") or cdir.name
            if name not in names:
                continue
            board = {}
            board_path = cdir / "identity_board.json"
            if board_path.is_file():
                board = json.loads(board_path.read_text())
            compact = {
                "name": name,
                "visual_description": data.get("visual_description"),
                "canonical_face": board.get("canonical_face"),
                "canonical_body": board.get("canonical_body"),
                "hair": board.get("hair"),
                "wardrobe": board.get("wardrobe") or data.get("wardrobe"),
                "palette": board.get("palette") or data.get("palette"),
                "do": board.get("do"),
                "dont": board.get("dont"),
            }
            entries.append({key: value for key, value in compact.items() if value})

        if not entries:
            return ""
        return (
            "## Bible character identity\n"
            "The video must keep each character matching this locked identity. Flag "
            "identity_drift on any mismatch.\n" + self._json_block(entries)
        )

    @staticmethod
    def _text(path) -> str:
        return path.read_text() if path.is_file() else ""

    @staticmethod
    def _json(path) -> Any:
        return json.loads(path.read_text()) if path.is_file() else {}

    def _shots(self, project) -> list[dict]:
        shots_path = project.path("storyboard", "shots.json")
        if not shots_path.is_file():
            return []
        return json.loads(shots_path.read_text()).get("shots", [])

    @staticmethod
    def _neighbors(shots: list[dict], shot_id: str) -> tuple[dict | None, dict | None]:
        for idx, shot in enumerate(shots):
            if shot.get("id") == shot_id:
                prev_shot = shots[idx - 1] if idx > 0 else None
                next_shot = shots[idx + 1] if idx + 1 < len(shots) else None
                return prev_shot, next_shot
        return None, None

    @staticmethod
    def _script_scene(script: dict, scene_no) -> dict:
        scenes = []
        for episode in script.get("episodes", []):
            scenes.extend(episode.get("scenes", []))
        for scene in scenes:
            if scene.get("scene") == scene_no:
                return scene
        return scenes[0] if len(scenes) == 1 else {}

    @staticmethod
    def _format_line(product_format) -> str:
        if not isinstance(product_format, dict):
            return "n/a"
        parts = [
            product_format.get("label") or product_format.get("name"),
            product_format.get("structure"),
            product_format.get("pacing"),
        ]
        return " | ".join(str(part) for part in parts if part) or "n/a"

    def _plot_summary(self, plot: dict) -> str:
        if not plot:
            return "(missing)"
        compact = {
            "logline": plot.get("logline"),
            "synopsis": plot.get("synopsis"),
            "themes": plot.get("themes", []),
            "characters": plot.get("characters", []),
        }
        return self._json_block(compact)

    @staticmethod
    def _compact_shot(shot: dict) -> dict:
        keep = (
            "id", "scene", "description", "camera", "camera_movement", "action",
            "composition", "emotion", "continuity_notes", "dialogue",
            "duration_s", "characters", "deps", "reference_characters",
        )
        return {key: shot.get(key) for key in keep if key in shot}

    def _neighbor_line(self, shot: dict | None) -> str:
        if not shot:
            return "none"
        compact = self._compact_shot(shot)
        return (
            f"{compact.get('id')}: "
            f"{compact.get('camera', '')} {compact.get('action') or compact.get('description', '')}"
        ).strip()

    @staticmethod
    def _json_block(data: Any) -> str:
        return "```json\n" + json.dumps(data, ensure_ascii=False, indent=2) + "\n```"

    @staticmethod
    def _hash(path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""
