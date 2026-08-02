"""Provider-aware preparation for paid image-generation requests.

Canonical project prompts remain untouched.  This module only creates a deterministic
provider projection when a selected image model declares a smaller hard prompt limit.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from .keyframe_prompt import STYLE_REFERENCE_GUARD, SUBJECT_REFERENCE_GUARD
from .providers.base import Generation, ImageCapabilities, ImageGen


_PROTECTED_BLOCKS = (SUBJECT_REFERENCE_GUARD, STYLE_REFERENCE_GUARD)
_SKILL_TITLES = {
    "character identity skill",
    "location identity skill",
    "micro-expression skill",
    "micro-expression performance skill",
    "image prompting skill",
    "cinematography skill",
    "storyboard grid (分镜图) prompting",
}
_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$")


class ImagePromptLimitError(ValueError):
    """A prompt cannot fit without removing unique project instructions."""


@dataclass(frozen=True)
class PreparedImagePrompt:
    canonical: str
    submitted: str
    measured_canonical: int
    measured_submitted: int
    limit: int | None
    unit: str
    removed_skill_sections: tuple[str, ...] = ()

    @property
    def was_fitted(self) -> bool:
        return self.submitted != self.canonical


def _measure(text: str, unit: str) -> int:
    if unit == "utf8_bytes":
        return len(text.encode("utf-8"))
    return len(text)


def _protect_blocks(text: str) -> tuple[str, dict[str, str]]:
    protected: dict[str, str] = {}
    for index, block in enumerate(_PROTECTED_BLOCKS):
        if block not in text:
            continue
        marker = f"# __STUDIO_PROTECTED_PROMPT_BLOCK_{index}__"
        text = text.replace(block, marker)
        protected[marker] = block
    return text, protected


def _expose_inline_skill_headings(text: str) -> str:
    for title in _SKILL_TITLES:
        pattern = re.compile(rf"[ \t]+(# {re.escape(title)})$", re.IGNORECASE | re.MULTILINE)
        text = pattern.sub(r"\n\1", text)
    return text


def _drop_skill_sections(text: str) -> tuple[str, tuple[str, ...]]:
    output: list[str] = []
    removed: list[str] = []
    skipped_heading_level: int | None = None

    for line in text.splitlines(keepends=True):
        heading = _HEADING.match(line.rstrip("\r\n"))
        if skipped_heading_level is not None:
            if heading is None or len(heading.group(1)) > skipped_heading_level:
                continue
            skipped_heading_level = None

        if heading is not None:
            title = heading.group(2).strip()
            if title.lower() in _SKILL_TITLES:
                removed.append(title)
                skipped_heading_level = len(heading.group(1))
                continue
        output.append(line)

    fitted = "".join(output).strip()
    fitted = re.sub(r"\n{3,}", "\n\n", fitted)
    return fitted, tuple(removed)


def prepare_image_prompt(
    prompt: str,
    capabilities: ImageCapabilities,
) -> PreparedImagePrompt:
    """Return the exact text safe to submit to the selected image model."""

    canonical = prompt.strip()
    unit = capabilities.prompt_length_unit
    limit = capabilities.max_prompt_length
    original_size = _measure(canonical, unit)
    if limit is None or original_size <= limit:
        return PreparedImagePrompt(
            canonical=canonical,
            submitted=canonical,
            measured_canonical=original_size,
            measured_submitted=original_size,
            limit=limit,
            unit=unit,
        )

    protected_text, protected = _protect_blocks(
        _expose_inline_skill_headings(canonical)
    )
    fitted, removed = _drop_skill_sections(protected_text)
    for marker, block in protected.items():
        fitted = fitted.replace(marker, block)
    fitted = fitted.strip()
    fitted_size = _measure(fitted, unit)
    if fitted_size > limit:
        raise ImagePromptLimitError(
            f"image prompt is {original_size} {unit}, exceeds {limit}, and cannot be "
            "fitted safely without removing unique project instructions"
        )
    return PreparedImagePrompt(
        canonical=canonical,
        submitted=fitted,
        measured_canonical=original_size,
        measured_submitted=fitted_size,
        limit=limit,
        unit=unit,
        removed_skill_sections=removed,
    )


def _projection_paths(out_path: str) -> tuple[Path, Path]:
    output = Path(out_path)
    return (
        output.with_suffix(".provider-prompt.md"),
        output.with_suffix(".provider-prompt.json"),
    )


def generate_project_image(
    project,
    image_provider: ImageGen,
    prompt: str,
    *,
    out_path: str,
    reference_images: list[str] | None = None,
    **kwargs: Any,
) -> Generation:
    """Validate, optionally fit, and submit one project image request."""

    prepared = prepare_image_prompt(prompt, image_provider.capabilities)
    project.assert_budget_available()
    prompt_path, metadata_path = _projection_paths(out_path)
    if prepared.was_fitted:
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(prepared.submitted)
        metadata_path.write_text(
            json.dumps(
                {
                    "provider": str(
                        getattr(image_provider, "name", type(image_provider).__name__)
                    ),
                    "model": str(getattr(image_provider, "model", "")),
                    "limit": prepared.limit,
                    "unit": prepared.unit,
                    "original_length": prepared.measured_canonical,
                    "submitted_length": prepared.measured_submitted,
                    "removed_skill_sections": list(
                        prepared.removed_skill_sections
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
    else:
        prompt_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)

    generation = image_provider.generate(
        prepared.submitted,
        out_path=out_path,
        reference_images=reference_images,
        **kwargs,
    )
    generation.meta = dict(generation.meta or {})
    generation.meta["prompt_projection"] = {
        "was_fitted": prepared.was_fitted,
        "limit": prepared.limit,
        "unit": prepared.unit,
        "original_length": prepared.measured_canonical,
        "submitted_length": prepared.measured_submitted,
        "submitted_prompt": prepared.submitted,
    }
    return generation
