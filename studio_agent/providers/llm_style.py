"""LLM-backed style profiler — the universal default for user-defined styles.

Turning a free-text style description into a structured, reusable style guide is a text
task, so any configured LLM can do it. This profiler is the default whenever no explicit
or vision style profiler is wired (e.g. a dashboard model-mix), so custom styles never
hard-fail for lack of a dedicated profiler.

It is text-only: when a reference image is supplied it cannot *see* it (the LLM interface
is text), so it profiles from the description and notes the image. The uploaded image
still rides downstream as a style-only reference independently of this profiler, so the
visual look propagates regardless. For true image style extraction, configure a vision
``style_profiler`` (see ``OpenAICompatibleStyleProfiler``).
"""

from __future__ import annotations

from pathlib import Path

from ..style import normalize_style_dict
from .base import Generation, LLM, StyleProfiler

_SYSTEM = (
    "You are a film art director who defines one consistent visual style for an entire "
    "AI film/series. Turn the user's brief into a reusable, model-agnostic style guide. "
    "Return strict JSON only."
)


class LLMStyleProfiler(StyleProfiler):
    """Profile a free-text style into a style dict via any configured LLM."""

    def __init__(self, llm: LLM, *, provider_name: str = "llm-style"):
        self.llm = llm
        self.name = provider_name

    def profile(
        self,
        *,
        description: str = "",
        image_path: str | None = None,
        language: str = "en",
        feedback: str = "",
    ) -> Generation:
        gen = self.llm.complete_json(self._prompt(description.strip(), image_path, language, feedback))
        style = normalize_style_dict(gen.content, description)
        return Generation(
            content=style,
            provider=gen.provider or self.name,
            model=gen.model,
            cost_usd=gen.cost_usd,
            seconds=gen.seconds,
            meta={"input": "style", "had_image": bool(image_path), "saw_image": False},
        )

    def _prompt(self, description: str, image_path: str | None, language: str, feedback: str = "") -> str:
        lines = ["[task:style_profile]"]
        if description:
            lines.append(f'The user describes the style as: "{description}".')
        if image_path:
            lines.append(
                f"A style reference image named '{Path(image_path).name}' was uploaded; "
                "honor any style cues implied by the description and the rest of the brief."
            )
        if not description and not image_path:
            lines.append("No explicit style was given; choose a tasteful, cohesive cinematic look.")
        if feedback:
            lines.append(
                f'The user reviewed the extracted style and asks for this targeted adjustment: "{feedback}". '
                "Apply only what this note changes and keep everything else identical."
            )
        lines.append(
            f"Write the style guide in language code '{language}'. Return JSON only with this "
            'shape: {"look": string (short style name), "label": string (concise handle), '
            '"palette": string, "aspect_ratio": string (e.g. "16:9"), "medium": string '
            "(2D/3D/2.5D, photographic, hand-drawn cel, stop-motion — name it explicitly), "
            '"idiom": string (artistic lineage, e.g. guoman, anime, Pixar, Ghibli, oil '
            'painting), "rendering": string, "lighting": string (hard/soft, key-fill, rim, '
            'volumetric, bloom, direction), "color_grade": string (warm/cool, saturation, '
            'contrast, filmic curve), "line_style": string, "lens": string (depth of field, '
            'bokeh, flare, vignette), "motion": string, "atmosphere": string (particles, '
            'haze, embers, glints), "notes": string, "prompt_playbook": string (a dense, '
            "model-optimized paragraph an image/video model can follow to reproduce this "
            "exact look on any new subject)}."
        )
        return "\n".join(lines)
