"""Stage contract + the provider bundle stages receive.

A stage is a pure-ish step: read prior files -> call LLM/providers -> write files ->
return a :class:`StageResult`. The state machine iterates stages uniformly through
this interface. ``Providers`` bundles the provider implementations chosen in
``config.yaml`` so stage code never imports a vendor SDK directly (invariant #6).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..providers.base import (
    LLM,
    ImageGen,
    Music,
    ReferenceAnalyzer,
    StyleProfiler,
    TTS,
    VideoGen,
    VLMCheck,
)


@dataclass
class Providers:
    """The provider implementations a run uses. Only ``llm`` is wired for the spine."""

    llm: LLM | None = None
    image: ImageGen | None = None
    video: VideoGen | None = None
    tts: TTS | None = None
    music: Music | None = None
    vlm: VLMCheck | None = None
    reference_analyzer: ReferenceAnalyzer | None = None
    style_profiler: StyleProfiler | None = None


@dataclass
class StageResult:
    status: str  # "complete" | "skipped" | "failed"
    message: str = ""


@dataclass
class StagePreflight:
    decision_required: bool = False
    request_path: str | None = None
    message: str = ""


class Stage(ABC):
    """One pipeline stage."""

    name: str

    def preflight(
        self,
        project,
        providers: Providers,
        *,
        auto: bool = False,
    ) -> StagePreflight:
        return StagePreflight()

    def on_approve(self, project, *, auto: bool = False) -> None:
        """Persist stage-specific approval artifacts before advancing."""
        return None

    @abstractmethod
    def run(self, project, providers: Providers) -> StageResult:
        ...
