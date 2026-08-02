"""Structured keyframe prompt compiler (invariant #8: prompts are artifacts).

A keyframe is a single still that anchors a shot, so this compiler gives the image model
strong *composition*, *lighting*, and *lens* direction plus identity conditioning — not
motion (that belongs to the video prompt). It is pure and deterministic: same shot +
style + references in, same markdown out. The storyboard stage owns reading/writing the
prompt file and preserving hand edits.
"""

from __future__ import annotations

import re
from typing import Any

from .style import format_style_prompt


_CAMERA_MOTION_WORDS = re.compile(
    r"\b(?:tracking|moving|dolly|push(?:-?in)?|pull(?:-?back)?|panning|tilting|"
    r"orbiting|crane|handheld)\b|\bthen\b",
    re.IGNORECASE,
)
_DYNAMIC_DESCRIPTION = re.compile(
    r"\b(?:walk|run|turn|open|ring|move|transform|become|enter|exit|cross|"
    r"raise|reach|grab|step|speak|say|fall|rise|chase|fight|jump|drive|fly)"
    r"(?:s|ed|ing)?\b|随后|然后|逐渐|走|跑|转身|变成|进入|离开",
    re.IGNORECASE,
)

# Still-image defects to steer away from (keyframes, not motion).
_AVOID_TERMS = (
    "text",
    "watermark",
    "logo",
    "deformed hands",
    "extra fingers",
    "extra limbs",
    "warped anatomy",
    "low detail",
    "blurry",
    "fused fingers",
    "malformed hands",
    "duplicated faces",
    "cloned faces",
    "floating objects",
    "unsupported objects",
    "impossible architecture",
    "inconsistent perspective",
    "melted textures",
    "disconnected limbs",
    "gibberish text",
)

# Style-only conditioning: when a project style reference image is supplied, take only its
# look from it, never its content. Shared by the keyframe and video prompt compilers.
STYLE_REFERENCE_GUARD = (
    "## Style reference\n"
    "A project style reference image is provided. Match only its visual style — medium, "
    "color palette, lighting, grain/texture, and overall rendering. Do not copy its "
    "subject, characters, or composition."
)
_STYLE_REFERENCE_GUARD = STYLE_REFERENCE_GUARD

SUBJECT_REFERENCE_GUARD = (
    "## Subject reference\n"
    "The provided character/location reference image(s) define this subject's IDENTITY ONLY "
    "— face, body, hair, distinguishing features, and wardrobe. Reproduce those faithfully "
    "and consistently. Do NOT copy the reference's art style, rendering, level of realism, "
    "lighting, or background: re-render the subject entirely in this project's defined art "
    "style and medium. If a reference is photographic or in a different style, convert it to "
    "the project's look while keeping the same identity and clothing."
)


def compile_keyframe_prompt(
    shot: dict[str, Any],
    *,
    ref_aliases: list[str] | None = None,
    location_aliases: list[str] | None = None,
    has_expression_sheet: bool = False,
    speaking: bool = False,
    style: dict[str, Any] | None = None,
    product_format: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
    prompt_skills: dict[str, str] | None = None,
    knowledge_guidance: str = "",
    hard_avoidances: list[str] | None = None,
    character_locks: dict[str, str] | None = None,
    character_rules: dict[str, dict[str, list[str]]] | None = None,
    has_style_reference: bool = False,
) -> str:
    """Compile an editable, composition-forward keyframe prompt for one shot."""
    style = style or {}
    context = context or {}
    prompt_skills = prompt_skills or {}
    aliases = [a for a in (ref_aliases or []) if a]
    locations = [a for a in (location_aliases or []) if a]

    camera = _static_camera(str(shot.get("camera") or "").strip())
    shot_size = str(shot.get("shot_size") or "").strip() or camera

    sections: list[str] = ["# Keyframe prompt"]

    comp = _composition(
        shot_size,
        str(shot.get("composition") or "").strip(),
        emotion=str(shot.get("emotion") or "").strip(),
    )
    sections.append("## Shot & composition\n" + comp)

    opening = _static_opening_moment(shot)
    moment = []
    description = str(shot.get("description") or "").strip()
    if (
        description
        and description.rstrip(" .").casefold() != opening.rstrip(" .").casefold()
        and not _DYNAMIC_DESCRIPTION.search(description)
    ):
        moment.append(description.rstrip(" ."))
    moment.append(opening)
    emotion = str(shot.get("emotion") or "").strip()
    if emotion:
        moment.append(f"Visible emotion: {emotion.rstrip(' .')}")
    sections.append(
        "## Subject & moment\n"
        + ". ".join(moment).rstrip(" .")
        + ". Render one sharp still image of this exact opening instant. Keep every subject "
        "in a clear, readable held pose with natural anatomy and precise spatial relationships."
    )

    opening_state = _opening_visual_state(shot)
    if opening_state:
        sections.append("## Opening visual state\n" + opening_state)

    world_context = _static_world_context(context)
    if world_context:
        sections.append("## Static world context\n" + world_context)

    continuity = _continuity_context(context)
    continuity_notes = str(shot.get("continuity_notes") or "").strip()
    if continuity_notes:
        continuity = (
            (continuity + "\n" if continuity else "")
            + f"Shot continuity notes: {continuity_notes}"
        )
    if continuity:
        sections.append("## Continuity\n" + continuity)

    sections.append(
        "## Lighting & lens\n"
        + _lighting_and_lens(
            shot_size,
            style,
            camera_angle=str(shot.get("camera_angle") or "").strip(),
            lens_intent=str(shot.get("lens_intent") or "").strip(),
        )
    )

    if aliases:
        who = ", ".join(aliases)
        sections.append(
            "## Identity & references\n"
            f"Featuring {who}. Use the provided reference images for exact identity — face, "
            "hair, wardrobe, and proportions must match."
        )

    locks = character_locks or {}
    lock_lines = [
        f"{alias}: {text.strip()}"
        for alias, text in locks.items()
        if text and text.strip()
    ]
    if lock_lines:
        sections.append(
            "## Character appearance lock (verbatim — do not alter)\n"
            + "\n".join(lock_lines)
        )

    rules_section = _character_rules_section(character_rules or {})
    if rules_section:
        sections.append(rules_section)

    if locations:
        where = ", ".join(locations)
        sections.append(
            "## Location & environment references\n"
            f"Set in {where}. Use the provided location reference images for architecture, "
            "materials, hero props, palette, and light state."
        )

    if speaking and has_expression_sheet:
        sections.append(
            "## Expression\n"
            "Match the provided facial expression reference sheet for the emotion of this beat."
        )

    if knowledge_guidance.strip():
        sections.append("## Retrieved filmmaking guidance\n" + knowledge_guidance.strip())

    skill_sections = _skill_sections(prompt_skills, speaking=speaking, has_locations=bool(locations))
    sections.extend(skill_sections)

    style_prompt = format_style_prompt(style)
    if style_prompt:
        sections.append("## Style\n" + style_prompt)

    if has_style_reference:
        sections.append(_STYLE_REFERENCE_GUARD)

    fmt_line = _format_direction(product_format)
    if fmt_line:
        sections.append("## Format\n" + fmt_line)

    avoid = list(_AVOID_TERMS) + [
        str(rule).strip() for rule in (hard_avoidances or []) if str(rule).strip()
    ]
    sections.append(
        "## Avoid\n" + ", ".join(dict.fromkeys(avoid))
        + ". One coherent frame, no collage or split panels."
    )
    return "\n\n".join(sections) + "\n"


def _character_rules_section(rules: dict[str, dict[str, list[str]]]) -> str:
    """Emit the identity board's do / don't / hero-prop / continuity rules per character.

    These are passed through verbatim for the model to honor — code never interprets their
    words (invariant #11). Empty when no rules exist so unrelated shots stay unchanged.
    """
    blocks: list[str] = []
    for alias, rule in rules.items():
        if not isinstance(rule, dict):
            continue
        do = [str(x).strip() for x in (rule.get("do") or []) if str(x).strip()]
        dont = [str(x).strip() for x in (rule.get("dont") or []) if str(x).strip()]
        props = [str(x).strip() for x in (rule.get("hero_props") or []) if str(x).strip()]
        priority = [str(x).strip() for x in (rule.get("continuity_priority") or []) if str(x).strip()]
        lines: list[str] = []
        if do:
            lines.append("  - Always: " + "; ".join(do))
        if dont:
            lines.append("  - Never: " + "; ".join(dont))
        if props:
            lines.append("  - Hero props (keep present and consistent): " + ", ".join(props))
        if priority:
            lines.append("  - Match first, in order: " + ", ".join(priority))
        if lines:
            blocks.append(f"{alias}:\n" + "\n".join(lines))
    if not blocks:
        return ""
    return (
        "## Character rules (keep these consistent)\n"
        + "\n".join(blocks)
    )


def _opening_visual_state(shot: dict[str, Any]) -> str:
    states = shot.get("start_states") or {}
    if not isinstance(states, dict) or not states:
        return ""
    lines = [
        "Start this still at exactly: "
        + ", ".join(f"{name}: {state}" for name, state in states.items())
        + "."
    ]
    lines.append(
        "Do not show later states or effects from later beats in this opening keyframe."
    )
    return "\n".join(lines)


def _skill_sections(
    prompt_skills: dict[str, str],
    *,
    speaking: bool,
    has_locations: bool,
) -> list[str]:
    sections: list[str] = []
    if prompt_skills.get("character_identity"):
        sections.append("## Character identity skill\n" + prompt_skills["character_identity"])
    if has_locations and prompt_skills.get("location_identity"):
        sections.append("## Location identity skill\n" + prompt_skills["location_identity"])
    if speaking and prompt_skills.get("micro_expression"):
        sections.append("## Micro-expression skill\n" + prompt_skills["micro_expression"])
    return sections


def _static_opening_moment(shot: dict[str, Any]) -> str:
    visual_beats = [
        beat for beat in (shot.get("visual_beats") or []) if isinstance(beat, dict)
    ]
    first = visual_beats[0] if visual_beats else {}
    for value in (
        shot.get("start_frame"),
        first.get("start_frame"),
        first.get("description"),
        shot.get("subject_blocking"),
        shot.get("description"),
    ):
        text = str(value or "").strip()
        if text:
            return text.rstrip(" .")
    return "the subject holds a clear, readable opening pose"


def _framing_for_size(shot_size: str) -> str:
    """Shot-size-specific framing guidance (structured cinematographic vocabulary)."""
    low = shot_size.lower()
    if "extreme close" in low or "ecu" in low:
        return "Fill the frame with the key detail; minimal headroom, the subject on the upper third."
    if "medium close" in low or "mcu" in low or "close" in low or low.strip() in {"cu"}:
        return "Hold the subject tight with controlled headroom and the eyes near the upper third."
    if "extreme wide" in low or "wide" in low or "establish" in low or "long" in low or "vista" in low:
        return "Let the environment dominate; place the subject small within deliberate negative space."
    if "over-the-shoulder" in low or "over the shoulder" in low or "ots" in low:
        return "Anchor the foreground shoulder low and soft; focus on the far subject's face."
    return "Balance the subject against the space with clear separation from the background."


def _composition(shot_size: str, composition: str = "", *, emotion: str = "") -> str:
    framing = shot_size or "balanced medium framing"
    base = (
        f"{framing}. {_framing_for_size(shot_size)} Compose with intent — clear focal point, "
        "deliberate use of the rule of thirds and depth, leading lines, and a "
        "foreground/midground/background read."
    )
    # Pass the beat's EMOTION (a still-safe state) through as a directive; never branch code
    # on its words (invariant #11), and never the action/dramatic_purpose (that is motion,
    # which belongs to the video prompt, not this still).
    beat = emotion.strip()
    if beat:
        base += (
            f" Let the framing express this beat: {beat.rstrip(' .')} — use balance, negative "
            "space, and subject placement to serve that emotion, not a neutral centered snapshot."
        )
    if composition:
        base += f" Specific layout: {composition}"
    return base


def _static_camera(camera: str) -> str:
    """Keep framing/lens language while stripping camera-path instructions."""
    static = _CAMERA_MOTION_WORDS.sub(" ", camera)
    static = " ".join(static.replace("--", " ").split()).strip(" ,-;")
    return static or "balanced medium framing"


def _focal_length(shot_size: str) -> str:
    """Map a shot-size description to a concrete lens focal length + depth-of-field intent.

    Order matters: the more specific sizes ("extreme close", "medium close") are matched
    before the looser "close"/"wide" keywords they contain.
    """
    low = shot_size.lower()
    if "extreme close" in low or "ecu" in low or "macro" in low or low.strip() == "ecu":
        return "a 100mm macro/telephoto lens with very shallow depth of field"
    if "medium close" in low or "mcu" in low:
        return "a 65mm lens with gentle subject-to-background separation"
    if "close" in low or low.strip() in {"cu"}:
        return "an 85mm portrait lens with shallow depth of field"
    if "extreme wide" in low or "wide" in low or "establish" in low or "long" in low or "vista" in low:
        return "a 24-35mm wide lens holding deep focus"
    if "over-the-shoulder" in low or "over the shoulder" in low or "ots" in low:
        return "a 50mm lens framed over the shoulder with the foreground shoulder soft"
    return "a 50mm lens at a natural perspective"


def _angle_phrase(camera_angle: str) -> str:
    low = camera_angle.lower()
    if "low" in low:
        return "shot from a low angle looking up to lend the subject scale and presence"
    if "high" in low or "overhead" in low or "top-down" in low or "bird" in low:
        return "shot from a high angle looking down"
    if "dutch" in low or "canted" in low or "tilted" in low:
        return "on a slight dutch/canted angle for unease"
    return "at eye level"


def _lighting_and_lens(
    shot_size: str,
    style: dict[str, Any],
    *,
    camera_angle: str = "",
    lens_intent: str = "",
) -> str:
    lens = _focal_length(shot_size)
    angle = _angle_phrase(camera_angle)
    palette = style.get("palette", "natural")
    lighting = str(style.get("lighting") or "").strip()
    grade = str(style.get("color_grade") or "").strip()
    if lighting:
        light_clause = f"Lighting: {lighting.rstrip(' .')}, motivated and modeling form ({palette})"
    else:
        light_clause = f"Cinematic, motivated lighting that models form and sets mood ({palette})"
    base = (
        f"{light_clause}; shoot on {lens}, {angle}. Use directional key light, controlled "
        "contrast, and a clear sense of depth."
    )
    if grade:
        base += f" Color grade: {grade.rstrip(' .')}."
    intent = str(lens_intent or "").strip()
    if intent:
        base += f" Lens intent: {intent.rstrip(' .')}."
    return base


def _format_direction(product_format: dict[str, Any] | None) -> str:
    if not product_format:
        return ""
    label = product_format.get("label") or product_format.get("name")
    return str(label) if label else ""


def _static_world_context(context: dict[str, Any]) -> str:
    parts: list[str] = []
    genre = str(context.get("genre") or "").strip()
    scene = context.get("scene") or {}
    if genre:
        parts.append(f"Genre/题材 (lock the world and look to this): {genre}")
    if scene.get("heading"):
        parts.append(f"Scene: {scene['heading']}")
    return "\n".join(f"- {part}" for part in parts)


def _continuity_context(context: dict[str, Any]) -> str:
    prev_line = _shot_frame_line(
        "Previous final frame",
        context.get("previous_shot"),
        field="end_frame",
    )
    next_line = _shot_frame_line(
        "Next opening frame",
        context.get("next_shot"),
        field="start_frame",
    )
    lines = [line for line in (prev_line, next_line) if line]
    if not lines:
        return ""
    lines.append(
        "Keep screen direction, wardrobe, props, lighting, and visible state consistent."
    )
    return "\n".join(lines)


def _shot_frame_line(
    label: str,
    shot: dict[str, Any] | None,
    *,
    field: str,
) -> str:
    if not shot:
        return ""
    frame = str(shot.get(field) or "").strip()
    if not frame:
        return ""
    sid = shot.get("id") or "adjacent"
    return f"{label} ({sid}): {frame}"
