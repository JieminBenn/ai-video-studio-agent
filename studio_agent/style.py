"""Style preset helpers for reproducible prompt artifacts."""

from __future__ import annotations

from dataclasses import dataclass


STYLE_FIELD_LABELS = {
    "look": "Look",
    "palette": "Palette",
    "aspect_ratio": "Aspect ratio",
    "medium": "Medium",
    "idiom": "Idiom",
    "rendering": "Rendering",
    "lighting": "Lighting",
    "color_grade": "Color grade",
    "line_style": "Line style",
    "lens": "Lens",
    "motion": "Motion",
    "atmosphere": "Atmosphere",
    "notes": "Notes",
}

STYLE_FIELD_ORDER = [
    "look",
    "palette",
    "aspect_ratio",
    "medium",
    "idiom",
    "rendering",
    "lighting",
    "color_grade",
    "line_style",
    "lens",
    "motion",
    "atmosphere",
    "notes",
]


# Sentinel name for a user-defined style. Unlike a preset, a custom style is not in
# ``style_presets`` — the style stage profiles the user's text/image into the dict and
# stores it on the project, so ``resolve_style`` returns an empty placeholder for it.
CUSTOM_STYLE_NAME = "custom"


@dataclass(frozen=True)
class ResolvedStyle:
    name: str
    style: dict


@dataclass(frozen=True)
class UnknownStyleError(ValueError):
    name: str
    available: list[str]


def resolve_style(config: dict, requested_name: str | None = None) -> ResolvedStyle:
    """Resolve a strict named style preset, with a legacy fallback for old configs.

    The ``custom`` sentinel is the user-defined path: it is not a preset, so it never
    raises ``UnknownStyleError`` and resolves to an empty placeholder the style stage
    fills in by profiling the user's text/image.
    """
    if requested_name == CUSTOM_STYLE_NAME:
        return ResolvedStyle(name=CUSTOM_STYLE_NAME, style={})

    presets = dict(config.get("style_presets") or {})
    if not presets:
        return ResolvedStyle(
            name=requested_name or config.get("default_style", "default"),
            style=dict(config.get("style") or {}),
        )

    name = requested_name or config.get("default_style") or next(iter(presets))
    if name not in presets:
        raise UnknownStyleError(name=name, available=sorted(presets))
    return ResolvedStyle(name=name, style=dict(presets[name] or {}))


def format_style_markdown(style: dict) -> str:
    """Return stable markdown bullet lines for a style dictionary."""
    lines = []
    for key in _ordered_keys(style):
        value = style.get(key)
        if value not in (None, ""):
            lines.append(f"- {STYLE_FIELD_LABELS.get(key, key.replace('_', ' ').title())}: {value}")
    return "\n".join(lines)


def style_medium_lead(style: dict) -> str:
    """A strong, leading art-style/medium instruction for image prompts.

    Image models weight the start of a prompt most heavily, and a "character model
    sheet / turnaround" framing biases them toward a glossy 3D design sheet. Surfacing
    the project's medium (e.g. ``2D手绘 / 国漫 / 数字绘画``) first — with an explicit
    do-not-substitute guard — keeps the rendering style from drifting (the reported
    2D->3D regression on regeneration). Medium-agnostic: the guard names both directions
    so a genuinely 3D project is unaffected.
    """
    descriptors: list[str] = []
    for key in ("medium", "look", "idiom", "rendering"):
        value = str(style.get(key) or "").strip()
        if value and value not in descriptors:
            descriptors.append(value)
    if not descriptors:
        return ""
    lead = ", ".join(descriptors)
    guard = (
        "Render strictly in this exact art style and medium, identical across every "
        "panel; do not substitute a different rendering technique (for example, do not "
        "turn a 2D / hand-drawn illustration into a 3D / CGI / photoreal render, or vice "
        "versa)."
    )
    return f"Art style and medium: {lead}. {guard}"


def format_style_prompt(style: dict) -> str:
    """Return a compact style sentence for image/video/audio prompts."""
    look = style.get("look", "cinematic")
    palette = style.get("palette", "natural")
    aspect = style.get("aspect_ratio")
    head = f"Style: {look}, {palette}"
    if aspect:
        head += f", aspect {aspect}"
    parts = [head + "."]

    for key in _ordered_keys(style):
        if key in {"look", "palette", "aspect_ratio"}:
            continue
        value = style.get(key)
        if value not in (None, ""):
            parts.append(f"{STYLE_FIELD_LABELS.get(key, key.replace('_', ' ').title())}: {value}.")
    return " ".join(parts)


# Style keys consumed elsewhere (not surface prompt text): the per-style prompt_playbook
# is for the image-prompt director only, so it must never leak into the compact style
# sentence or bible/style.md.
_NON_PROMPT_FIELDS = {"prompt_playbook"}

# Keys a profiler must fill so downstream threading matches the preset shape.
STYLE_PROFILE_KEYS = (
    "look", "palette", "aspect_ratio", "medium", "idiom", "rendering",
    "lighting", "color_grade", "line_style", "lens", "motion", "atmosphere", "notes",
)


def normalize_style_dict(value: object, description: str = "") -> dict:
    """Coerce a profiler's JSON into a complete style dict (shared by all profilers).

    Guarantees every key the pipeline reads is present so a sparse or malformed model
    response never breaks downstream prompt threading. ``label`` is a short human-facing
    handle; ``prompt_playbook`` is the director-only idiom.
    """
    data = value if isinstance(value, dict) else {}
    style: dict = {key: str(data.get(key) or "").strip() for key in STYLE_PROFILE_KEYS}
    style["look"] = style["look"] or (description.strip() or "cinematic")
    style["aspect_ratio"] = style["aspect_ratio"] or "16:9"
    style["prompt_playbook"] = str(data.get("prompt_playbook") or "").strip()
    label = str(data.get("label") or "").strip() or style["look"]
    style["label"] = " ".join(label.split()[:8])
    return style


def _ordered_keys(style: dict) -> list[str]:
    known = [key for key in STYLE_FIELD_ORDER if key in style]
    extra = sorted(
        key for key in style if key not in STYLE_FIELD_ORDER and key not in _NON_PROMPT_FIELDS
    )
    return known + extra
