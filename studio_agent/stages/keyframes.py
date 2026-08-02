"""Approved keyframe prompt batch -> generated still images.

The planning stages persist editable prompts.  This stage is intentionally the only
place where those prompts cross the image-provider boundary, after their file-backed
approval manifest has been verified.
"""

from __future__ import annotations

import json
from pathlib import Path

from .base import Providers, StageResult
from .storyboard import StoryboardStage
from ..asset_regeneration import (
    pending_regeneration,
    promote_candidate,
    record_regeneration_error,
)
from ..image_generation import generate_project_image
from ..invalidation import invalidate_shots
from ..prompt_approvals import confirm_prompt, prompt_path, require_prompt_batch_approval
from ..reference_assets import cap_reference_paths, live_reference_paths_for_shot


class KeyframesStage(StoryboardStage):
    """Generate only missing images from the exact approved prompt files."""

    name = "keyframes"

    def run(self, project, providers: Providers) -> StageResult:
        if project.stage_status(self.name) == "complete":
            return StageResult(status="skipped", message="keyframes already complete")

        require_prompt_batch_approval(project, "keyframes")
        shots_path = project.path("storyboard", "shots.json")
        shots = json.loads(shots_path.read_text()).get("shots") or []

        for shot in shots:
            request = pending_regeneration(project, str(shot["id"]), "keyframe")
            if request is not None:
                self._render_candidate(project, shot, request, providers)
                continue
            if isinstance(shot.get("motion_grid"), dict):
                if not self._grid_active(project, providers):
                    raise ValueError(
                        "motion-grid keyframe generation requires grid-capable "
                        "image and video providers"
                    )
                self._render_grid_keyframe(
                    project,
                    shot,
                    providers,
                    prepared_only=True,
                )
            else:
                self._render_missing_keyframes(project, [shot], providers)

        return StageResult(status="complete", message=f"keyframes: {len(shots)} image(s)")

    def _render_candidate(self, project, shot: dict, request: dict, providers: Providers) -> None:
        shot_id = str(shot["id"])
        if isinstance(shot.get("motion_grid"), dict) and not self._grid_active(project, providers):
            raise ValueError(
                "motion-grid keyframe generation requires grid-capable image and video providers"
            )
        candidate_root = project.path(
            "assets", ".candidates", str(request["request_id"])
        )
        candidate = candidate_root / str(shot["keyframe"])
        live_rel = f"storyboard/keyframes/{shot['keyframe']}"
        live = project.path(*live_rel.split("/"))
        reference_fields = (
            ("keyframe_reference_images",)
            if "keyframe_reference_images" in shot
            else ("reference_images",)
        )
        rel_refs = live_reference_paths_for_shot(project, shot, reference_fields)
        abs_refs = [str(project.dir / ref) for ref in rel_refs]
        abs_refs = [ref for ref in abs_refs if Path(ref).is_file()]
        named_rel = set(shot.get("named_reference_images") or [])
        named_refs = [
            str(project.dir / ref)
            for ref in rel_refs
            if ref in named_rel and str(project.dir / ref) in abs_refs
        ]
        abs_refs = cap_reference_paths(
            project,
            abs_refs,
            providers.image.capabilities.max_reference_images,
            named_refs=named_refs,
        )
        try:
            gen = generate_project_image(
                project,
                providers.image,
                prompt_path(project, "keyframes", shot).read_text(),
                out_path=str(candidate),
                reference_images=abs_refs,
                seed=shot.get("reference_seed", 0),
            )
            project.add_generation_cost(stage=self.name, generation=gen)
            live_family = [
                live_rel,
                live.with_suffix(".provider-prompt.md").relative_to(project.dir).as_posix(),
                live.with_suffix(".provider-prompt.json").relative_to(project.dir).as_posix(),
            ]
            invalidate_shots(
                project,
                [shot_id],
                from_stage="keyframes",
                reason="keyframe-candidate-ready",
                source_path=live_rel,
                include_keyframes=True,
                preserve_paths=live_family,
            )
            moves = {candidate: live_rel}
            for suffix in (".provider-prompt.md", ".provider-prompt.json"):
                sidecar = candidate.with_suffix(suffix)
                if sidecar.is_file():
                    moves[sidecar] = live.with_suffix(suffix).relative_to(project.dir).as_posix()
            promote_candidate(
                project,
                request,
                candidate_to_live=moves,
                live_family=live_family,
            )
            confirm_prompt(project, "keyframes", shot_id, confirmer="regeneration")
        except Exception as exc:
            record_regeneration_error(project, shot_id, "keyframe", exc)
            raise
