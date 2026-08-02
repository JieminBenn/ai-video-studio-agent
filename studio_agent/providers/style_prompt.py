"""Shared style-profiling prompt for every vision style profiler.

The style system prompt and the user-message JSON shape are identical across the
OpenAI-compatible, Gemini, and Anthropic style profilers, so they live here once. Each
profiler imports ``STYLE_SYSTEM`` and ``style_user_text`` and differs only in how it ships
the image + text to its vendor SDK.
"""

from __future__ import annotations

STYLE_SYSTEM = (
    "You are a film art director who defines one consistent visual style for an entire "
    "AI film/series. Given a written description and/or a reference image, you extract a "
    "reusable, model-agnostic style guide. When given an image, describe ONLY its visual "
    "style — its medium (2D/3D/photographic), artistic idiom, color palette, lighting, "
    "color grade, rendering, lens/grain, atmosphere, and texture — and explicitly IGNORE "
    "its subjects, characters, and composition. When the image and the text disagree on "
    "the medium or rendering, the IMAGE is authoritative for visual mechanics; the text "
    "only informs naming and mood. Return strict JSON only."
)


def style_user_text(description: str, has_image: bool, language: str, feedback: str = "") -> str:
    source = []
    if description:
        source.append(f'The user describes the style as: "{description}".')
    if has_image:
        source.append(
            "A reference image is attached. Read ONLY its visual style (medium, idiom, "
            "palette, lighting, color grade, rendering, lens/grain, atmosphere, texture, mood); never describe "
            "or reuse its subjects, characters, or composition."
        )
    if feedback:
        source.append(
            f'The user reviewed the extracted style and asks for this targeted adjustment: "{feedback}". '
            "Apply only what this note changes and keep everything else identical."
        )
    return (
        " ".join(source)
        + f" Write the style guide in language code '{language}'. Return JSON only with "
        'this shape: {"look": string (a short style name), "label": string (a concise '
        'human-facing handle), "palette": string, "aspect_ratio": string (e.g. "16:9"), '
        '"medium": string (2D/3D/2.5D, photographic, hand-drawn cel, stop-motion — name '
        'it explicitly), "idiom": string (artistic lineage, e.g. guoman, anime, Pixar, '
        'Ghibli, oil painting), "rendering": string, "lighting": string (hard/soft, '
        'key-fill, rim, volumetric, bloom, direction), "color_grade": string (warm/cool, '
        'saturation, contrast, filmic curve), "line_style": string, "lens": string '
        '(depth of field, bokeh, flare, vignette), "motion": string, "atmosphere": string '
        '(particles, haze, embers, glints), "notes": string, "prompt_playbook": string '
        "(a dense, model-optimized paragraph an image/video model can follow to reproduce "
        "this exact look on any new subject)}."
    )
