"""Assemble stage: build the EDL and render the final cut (timeline.json -> mp4).

The last M0 stage. It builds ``edit/timeline.json`` from the clip/audio manifests, then
renders ``output/<project-id>.mp4`` via an injected renderer (FFmpeg by default; tests
inject a fake). The render is the one expensive, environment-dependent step, so it lives
behind the renderer seam while the EDL building stays pure and deterministic.

Idempotent (invariant #3): skips if complete, and won't re-render when the output mp4
already exists.
"""

from __future__ import annotations

import time

from ..assembly.ffmpeg_edit import FFmpegRenderer, build_timeline
from .base import Providers, Stage, StageResult


class AssembleStage(Stage):
    name = "assemble"

    def __init__(self, renderer=None):
        self.renderer = renderer or FFmpegRenderer()

    def run(self, project, providers: Providers) -> StageResult:
        if project.stage_status(self.name) == "complete":
            return StageResult(status="skipped", message="assemble already complete")

        timeline = build_timeline(project)
        out = project.path("output", f"{project.project_id}.mp4")

        if out.is_file():
            return StageResult(status="complete", message=f"{out.name} already rendered")

        start = time.time()
        self.renderer.render(project, timeline, str(out))
        project.add_cost(stage=self.name, provider="ffmpeg",
                         cost_usd=0.0, seconds=round(time.time() - start, 2))
        return StageResult(status="complete", message=f"rendered {out.name}")
