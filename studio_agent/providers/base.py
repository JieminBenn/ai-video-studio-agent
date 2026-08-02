"""Provider abstraction (invariant #6: provider-agnostic).

Every model call in the pipeline goes through one of these interfaces. Stage code
never imports a vendor SDK or hardcodes a model name — it asks for a provider via
``config.yaml`` and calls the interface. Swapping a model or self-hosting later is a
new implementation behind the same interface, not a pipeline rewrite.

Each call returns its content alongside a :class:`Generation` record so the
orchestrator can log cost + time per generation (invariant #7).

Only :class:`LLM` is *implemented* in the foundational spine (see ``fake.py``).
``ImageGen`` / ``VideoGen`` / ``TTS`` / ``Music`` / ``VLMCheck`` are the contracts the
later M0 stages will implement; they are declared here so the abstraction is fixed
from the start.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Generation:
    """One generation's content plus its cost/time record.

    ``cost_usd`` and ``seconds`` flow into ``project.json``'s cost log. Fakes report
    zero cost but a realistic-looking record so the logging path is exercised offline.
    """

    content: Any
    provider: str
    model: str
    cost_usd: float = 0.0
    seconds: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


class ManualImportRequired(RuntimeError):
    """A manual provider wrote instructions and needs the user to import an artifact."""

    def __init__(
        self,
        *,
        provider: str,
        out_path: str,
        request_path: str,
        message: str | None = None,
    ):
        self.provider = provider
        self.out_path = out_path
        self.request_path = request_path
        super().__init__(
            message
            or (
                f"{provider} requires manual import. Follow {request_path}, "
                f"save the result to {out_path}, then resume."
            )
        )


@dataclass(frozen=True)
class ImageCapabilities:
    """What an image provider can do that stage logic must gate on.

    ``supports_storyboard_grid`` marks models verified to render clean, consistent
    multi-panel contact-sheet grids (set per-profile in config). Default False so the
    grid feature only activates on an explicit allowlist (invariant #6).
    """

    supports_storyboard_grid: bool = False
    max_reference_images: int = 9
    max_prompt_length: int | None = None
    prompt_length_unit: str = "characters"

    def __post_init__(self) -> None:
        if self.prompt_length_unit not in {"characters", "utf8_bytes"}:
            raise ValueError(
                "image prompt length unit must be 'characters' or 'utf8_bytes'"
            )
        if self.max_prompt_length is not None and self.max_prompt_length <= 0:
            raise ValueError("image max prompt length must be positive")


class LLM(ABC):
    """Text/structured generation for creative + structuring steps (plot, script)."""

    @abstractmethod
    def complete(self, prompt: str, *, system: str | None = None) -> Generation:
        """Return free-form text for ``prompt``."""

    @abstractmethod
    def complete_json(self, prompt: str, *, system: str | None = None) -> Generation:
        """Return parsed JSON (``Generation.content`` is a dict/list)."""


class ImageGen(ABC):
    """Reference-conditioned image generation (bible references, keyframes)."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        out_path: str,
        reference_images: list[str] | None = None,
        **kwargs: Any,
    ) -> Generation:
        ...

    @property
    def capabilities(self) -> ImageCapabilities:
        return ImageCapabilities(
            supports_storyboard_grid=getattr(self, "_supports_storyboard_grid", False),
            max_reference_images=getattr(self, "max_reference_images", 9),
            max_prompt_length=getattr(self, "max_prompt_length", None),
            prompt_length_unit=getattr(self, "prompt_length_unit", "characters"),
        )


class ReferenceAnalyzer(ABC):
    """Classify uploaded reference images before they have a concrete bible target."""

    @abstractmethod
    def analyze(
        self,
        image_path: str,
        *,
        aliases: list[str] | None = None,
        user_note: str = "",
    ) -> Generation:
        """Return target_type/target_id/confidence metadata for an uploaded image."""

    def describe(
        self,
        image_paths: list[str],
        *,
        prompt: str,
        language: str = "en",
    ) -> Generation:
        """Return a JSON visual identity (face/body/hair/wardrobe/palette/...) read directly
        from the reference image(s), grounding the bible in the upload instead of inventing
        it from text. Analyzers without vision-grounded description raise NotImplementedError
        so callers fall back to the text path."""
        raise NotImplementedError("this reference analyzer cannot describe images")

    def revise(
        self,
        image_paths: list[str],
        *,
        prompt: str,
        language: str = "en",
    ) -> Generation:
        """Return a revised artifact as RAW TEXT, grounded in the reference image(s).

        Used by regeneration so the model can SEE the named reference and apply the user's
        instruction faithfully. Analyzers without vision raise NotImplementedError so callers
        fall back to the text path."""
        raise NotImplementedError("this reference analyzer cannot revise with images")


class StyleProfiler(ABC):
    """Turn a user's free-form style input into a reusable style dict.

    The user defines the project's look by typing a label/description, writing a
    sentence, or uploading a reference image whose *style only* (palette, lighting,
    rendering medium, lens/grain, texture, mood) is extracted — never its subject or
    composition. The result is the same dict shape the fixed ``style_presets`` used
    (``look``/``palette``/``aspect_ratio``/``rendering``/``line_style``/``motion``/
    ``prompt_playbook`` + a short ``label``), so it threads through every downstream
    prompt with no other change.
    """

    @abstractmethod
    def profile(
        self,
        *,
        description: str = "",
        image_path: str | None = None,
        language: str = "en",
        feedback: str = "",
    ) -> Generation:
        """Return a style dict in ``Generation.content`` from text and/or an image.

        ``feedback`` (optional) is a user note refining a prior extraction — apply only what
        it targets and keep the rest of the look unchanged.
        """


@dataclass(frozen=True)
class VideoCapabilities:
    """What reference inputs a video provider can accept for one generation.

    The prompt compiler reads these so it never instructs the model to use inputs the
    selected provider will silently drop. For example, the BytePlus ModelArk Seedance
    profile currently must not mix a first-frame keyframe with extra reference media,
    so its capabilities report no reference-image and no last-frame support.
    """

    supports_reference_images: bool = True
    supports_last_frame: bool = True
    supports_native_audio: bool = False
    max_image_inputs: int = 9
    supports_storyboard_grid: bool = False
    min_duration_s: int = 1
    max_duration_s: int = 15
    # The provider's own default clip length. ``None`` means the adapter can omit the
    # duration entirely so the model's native default governs — callers signal that by
    # passing ``duration_s=None`` to ``VideoGen.generate()``.
    default_duration_s: int | None = None


class VideoGen(ABC):
    """Image-to-video / text-to-video clip generation."""

    @abstractmethod
    def generate(self, prompt: str, *, out_path: str, **kwargs: Any) -> Generation:
        ...

    @property
    def capabilities(self) -> VideoCapabilities:
        """Reference/continuity inputs this provider accepts (default: full support)."""
        return VideoCapabilities()


class TTS(ABC):
    """Text-to-dialogue / voice synthesis."""

    @abstractmethod
    def speak(self, text: str, *, out_path: str, **kwargs: Any) -> Generation:
        ...


class Music(ABC):
    """Music / score generation."""

    @abstractmethod
    def compose(self, prompt: str, *, out_path: str, **kwargs: Any) -> Generation:
        ...


# The QC dimensions every generated clip is checked against (invariant #9). Story and
# shot intent come first so regeneration decisions are not dominated by surface polish.
AUDIO_QC_DIMENSIONS = (
    "dialogue_accuracy",
    "music_absence",
    "sound_design",
    "audio_sync",
)

QC_DIMENSIONS = [
    "story_alignment",
    "shot_instruction_adherence",
    "continuity",
    "identity_drift",
    "motion_anatomy",
    "artifacts",
    *AUDIO_QC_DIMENSIONS,
    "safety",
]

QC_BLOCKING_DIMENSIONS = {
    "story_alignment",
    "shot_instruction_adherence",
    "continuity",
    "identity_drift",
    "motion_anatomy",
    "dialogue_accuracy",
    "music_absence",
    "audio_sync",
    "safety",
}

QC_SEVERITY_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}


def qc_failure_blocks(check: dict[str, Any]) -> bool:
    """Whether a failed QC check should force regeneration.

    Low-severity visual artifacts are warnings: they should be visible in the report,
    but not block story review or automatically spend on regeneration. Story/shot
    mismatch, continuity breaks, identity drift, broken motion, audio mismatch, and
    safety failures remain blocking whenever the VLM marks them failed.
    """
    if check.get("passed", True):
        return False
    dimension = str(check.get("dimension") or "")
    severity = str(check.get("severity") or "high")
    if dimension in {"artifacts", "sound_design"}:
        return QC_SEVERITY_RANK.get(severity, 0) >= QC_SEVERITY_RANK["medium"]
    return dimension in QC_BLOCKING_DIMENSIONS


def qc_report_status(checks: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize per-dimension checks into stable report-level status fields."""
    failures = [check for check in checks if not check.get("passed", True)]
    blocking = [check for check in failures if qc_failure_blocks(check)]
    max_severity = max(
        (str(check.get("severity") or "none") for check in failures),
        key=lambda severity: QC_SEVERITY_RANK.get(severity, 0),
        default="none",
    )
    overall_pass = not blocking
    has_warnings = bool(failures) and overall_pass
    return {
        "overall_pass": overall_pass,
        "max_severity": max_severity if failures else "none",
        "recommendation": (
            "regenerate" if blocking
            else "approve_with_warnings" if has_warnings
            else "approve"
        ),
        "has_warnings": has_warnings,
        "blocking_failures": [check.get("dimension") for check in blocking],
    }


class VLMCheck(ABC):
    """Automated visual/audio QC of a generated clip (invariant #9)."""

    @property
    def supports_audio_review(self) -> bool:
        return False

    @abstractmethod
    def review(self, clip_path: str, *, prompt: str, **kwargs: Any) -> Generation:
        ...
