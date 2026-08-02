"""Deterministic 分镜图 (storyboard motion-grid) prompt compiler (invariant #8).

Mirrors keyframe_prompt / video_prompt: pure, no I/O, no provider calls, fully
unit-testable. Emits a structured brief that the LLM prompt director rewrites into the
dense contact-sheet prompt actually sent to the image model. The brief encodes the hard
constraints that make a grid clean and followable by Seedance: exact R x C layout, one
consistent character/location across panels, and an ordered per-panel motion progression.
"""

from __future__ import annotations

from typing import Any

from .style import format_style_prompt

_ALIASES: dict[str, tuple[int, int]] = {
    "九宫格": (3, 3),
    "十二宫格": (3, 4),
    "2x2": (2, 2),
    "3x3": (3, 3),
    "3x4": (3, 4),
    "2x3": (2, 3),
}


def resolve_layout(value: str, duration_s: float = 4.0) -> tuple[int, int, int]:
    key = (value or "auto").strip()
    if key in _ALIASES:
        rows, cols = _ALIASES[key]
        return rows, cols, rows * cols
    if key.lower() in _ALIASES:
        rows, cols = _ALIASES[key.lower()]
        return rows, cols, rows * cols
    # auto / unknown: scale panel density with the clip's length.
    try:
        seconds = float(duration_s)
    except (TypeError, ValueError):
        seconds = 4.0
    rows, cols = (2, 2) if seconds <= 6 else (3, 3)
    return rows, cols, rows * cols


def panel_beats(
    action: str,
    panel_count: int,
    *,
    visual_beats: list[dict[str, Any]] | None = None,
) -> list[str]:
    ordered = [beat for beat in (visual_beats or []) if isinstance(beat, dict)]
    if ordered:
        beats = []
        for i in range(panel_count):
            source_index = min((i * len(ordered)) // panel_count, len(ordered) - 1)
            source = ordered[source_index]
            detail = str(
                source.get("start_frame")
                or source.get("description")
                or source.get("end_frame")
                or source.get("id")
                or ""
            ).strip()
            state = source.get("end_states") or {}
            state_text = ", ".join(f"{name}: {value}" for name, value in state.items())
            suffix = f" End state: {state_text}." if state_text else ""
            beats.append(
                f"Panel {i + 1} — frozen still: {detail.rstrip(' .')}.{suffix}"
            )
        return beats
    beats: list[str] = []
    for i in range(panel_count):
        if i == 0:
            desc = "the subject holds the opening pose in the established composition"
        elif i == panel_count - 1:
            desc = "the subject holds the clean final pose"
        else:
            desc = f"the subject holds discrete intermediate pose {i + 1}"
        beats.append(f"Panel {i + 1} — frozen still: {desc}")
    return beats


def compile_grid_prompt(
    shot: dict[str, Any],
    *,
    rows: int,
    cols: int,
    style: dict[str, Any] | None = None,
    product_format: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
    prompt_skills: dict[str, str] | None = None,
    character_locks: list[str] | None = None,
) -> str:
    style = style or {}
    prompt_skills = prompt_skills or {}
    n = rows * cols
    action = str(shot.get("action") or "").strip()
    description = str(shot.get("description") or "").strip()
    characters = [str(c).strip() for c in (shot.get("characters") or []) if str(c).strip()]
    locations = [str(c).strip() for c in (shot.get("reference_locations") or []) if str(c).strip()]
    subject = ", ".join(characters) if characters else "the subject"
    place = ", ".join(locations) if locations else "the same location"

    sections: list[str] = [f"# Storyboard grid brief — {shot.get('id', 'shot')}"]

    sections.append(
        "## Grid layout\n"
        f"One single composite storyboard contact sheet of exactly {n} frames arranged in "
        f"a strict {rows} rows by {cols} columns grid in row-major order: left-to-right, "
        "top-to-bottom. "
        "Uniform panel size; thin clean gutters between panels; each panel is a 16:9 frame. "
        "No panel numbers, no captions, no text, no decorative borders."
    )

    sections.append(
        "## Consistency\n"
        f"Every panel shows the SAME character identity ({subject}) and the SAME location ({place}). "
        "Keep recognizable face structure, identity traits, spatial continuity, palette, lighting, "
        "and art style locked. Allow only the intentional state changes defined below; do not create "
        "accidental face, wardrobe, or anatomy drift between panels."
    )

    locks = [str(text).strip() for text in (character_locks or []) if str(text).strip()]
    if locks:
        sections.append(
            "## Character appearance lock (verbatim — do not alter)\n"
            + "\n".join(locks)
        )

    subject_lines = [description.rstrip(" .")] if description else []
    start_frame = str(shot.get("start_frame") or "").strip()
    if start_frame and start_frame.rstrip(" .").lower() not in {
        line.lower() for line in subject_lines
    }:
        subject_lines.append(start_frame.rstrip(" ."))
    if subject_lines:
        sections.append(
            "## Static subject setup\n"
            + ". ".join(subject_lines).rstrip(" .")
            + "."
        )

    beats = panel_beats(action, n, visual_beats=shot.get("visual_beats"))
    sections.append(
        "## Panel sequence\n"
        + "\n".join(f"- {b}" for b in beats)
        + f"\nTogether the {n} frozen panels form one ordered visual progression."
    )

    start_states = shot.get("start_states") or {}
    end_states = shot.get("end_states") or {}
    if start_states or end_states:
        start_text = ", ".join(f"{name}: {value}" for name, value in start_states.items()) or "unchanged"
        end_text = ", ".join(f"{name}: {value}" for name, value in end_states.items()) or "unchanged"
        sections.append(
            "## State trajectory\n"
            f"Opening: {start_text}. Final: {end_text}. Preserve identity through all intentional state changes."
        )

    style_prompt = format_style_prompt(style)
    if style_prompt:
        sections.append("## Style\n" + style_prompt)

    if prompt_skills.get("character_identity"):
        sections.append("## Character identity skill\n" + prompt_skills["character_identity"])

    sections.append(
        "## Avoid\n"
        "collage, scrapbook, poster, irregular or uneven grid, mismatched character across "
        "panels, drifting wardrobe or lighting, captions, watermark, panel numbers, thick "
        "decorative borders, a single non-grid image, "
        "fused fingers, malformed hands, extra or missing fingers, duplicated or warped faces, "
        "floating objects, unsupported objects, impossible architecture, disconnected limbs, "
        "melted textures, gibberish text."
    )

    return "\n\n".join(sections) + "\n"
