"""Provider-aware video prompt compiler (invariant #8: prompts are artifacts).

Stage code asks this layer for a compiled, editable video prompt instead of hand-rolling
a flat sentence. The goal is prompts that drive *actual motion* — clear subject action,
temporal beats across the shot, camera movement, foreground/background life, and acting
direction — while respecting what the selected provider can actually accept (reference
images, last-frame carry) via :class:`~studio_agent.providers.base.VideoCapabilities`.

The compiler is pure and deterministic: same shot + style + format + capabilities in,
same markdown out. It performs no I/O and calls no provider, so it is unit-testable
offline. The video stage owns reading/writing the prompt file and preserving hand edits.
"""

from __future__ import annotations

import math
from typing import Any

from .cinematography_recipes import fill_recipe_slots
from .keyframe_prompt import STYLE_REFERENCE_GUARD
from .providers.base import VideoCapabilities
from .style import format_style_prompt

# Framing text that already implies camera motion — don't bolt another move on top.
_MOVEMENT_WORDS = (
    "pan", "tilt", "dolly", "push", "pull", "track", "zoom", "handheld",
    "orbit", "crane", "truck", "follow", "whip", "arc", "steadicam", "drift",
)

# Language that signals a dead, non-moving frame — explicitly disallow it.
_AVOID_TERMS = (
    "static pose",
    "still image",
    "frozen frame",
    "slideshow",
    "motionless portrait",
    "lifeless stare",
    "a photo that does not move",
)


def compile_video_prompt(
    shot: dict[str, Any],
    *,
    style: dict[str, Any] | None = None,
    product_format: dict[str, Any] | None = None,
    capabilities: VideoCapabilities | None = None,
    has_previous_shot: bool = False,
    context: dict[str, Any] | None = None,
    prompt_skills: dict[str, str] | None = None,
    camera_recipe: dict[str, Any] | None = None,
    knowledge_guidance: str = "",
    hard_avoidances: list[str] | None = None,
    has_style_reference: bool = False,
    language: str = "en",
) -> str:
    """Compile an editable, motion-forward video prompt for one shot.

    ``shot`` provides camera/action/description/dialogue/duration plus character and
    reference metadata; ``style`` and ``product_format`` add the project's look and
    format intent; ``capabilities`` gates continuity/reference language to what the
    provider supports; ``has_previous_shot`` enables last-frame carry continuity.
    """
    caps = capabilities or VideoCapabilities()
    style = style or {}
    context = context or {}
    prompt_skills = prompt_skills or {}

    duration = _duration(shot.get("duration_s"))
    camera = str(shot.get("camera") or "").strip()
    camera_movement = str(shot.get("camera_movement") or "").strip()
    # When no explicit movement is set, fall back to the recipe's short name so the
    # base camera line and motion beats stay in sync with the recipe's detailed text.
    recipe_name = str((camera_recipe or {}).get("name") or "").strip()
    effective_movement = camera_movement or recipe_name
    action = str(shot.get("action") or "").strip()
    description = str(shot.get("description") or "").strip()
    composition = str(shot.get("composition") or "").strip()
    emotion = str(shot.get("emotion") or "").strip()
    continuity_notes = str(shot.get("continuity_notes") or "").strip()
    characters = [str(c).strip() for c in (shot.get("characters") or []) if str(c).strip()]
    locations = [
        str(c).strip() for c in (shot.get("reference_locations") or []) if str(c).strip()
    ]
    dialogue = [
        str(d.get("line", "")).strip()
        for d in (shot.get("dialogue") or [])
        if str(d.get("line", "")).strip()
    ]
    subject = characters[0] if characters else "the subject"
    is_static = str((camera_recipe or {}).get("motion_strength") or "").strip() == "static"

    sections: list[str] = [f"# Video prompt — {shot.get('id', 'shot')}"]

    story_beat = _story_beat(context)
    if story_beat:
        sections.append("## Story beat\n" + story_beat)

    performance = _performance_objective(shot, context, subject)
    if performance:
        sections.append("## Performance objective\n" + performance)

    action_clause = action.rstrip(" .")
    subject_action = action_clause or f"{subject} moves through the scene"
    subject_lines = [description.rstrip(" .")] if description else []
    subject_lines.append(subject_action)
    sections.append("## Subject & action\n" + ". ".join(subject_lines).rstrip(" .") + ".")

    state_trajectory = _state_trajectory(shot)
    if state_trajectory:
        sections.append("## State trajectory\n" + state_trajectory)

    target_state = _target_state_direction(shot, caps)
    if target_state:
        sections.append("## Target state lock\n" + target_state)

    grid = shot.get("motion_grid") or {}
    if grid:
        n = grid.get("panel_count") or (int(grid.get("rows", 0)) * int(grid.get("cols", 0)))
        rows, cols = grid.get("rows"), grid.get("cols")
        layout = grid.get("layout") or (f"{rows}x{cols}" if rows and cols else f"{n}-panel")
        sections.append(
            "## Storyboard grid\n"
            f"The supplied image is a single {n}-panel storyboard grid ({layout}). Generate "
            "ONE continuous shot that follows the panels in order, left-to-right then "
            "top-to-bottom, interpolating smooth motion between them. Do not show the grid, "
            "the gutters, or multiple frames in the output — render only the live scene."
        )

    beats_header = (
        f"## Motion beats (~{_fmt(duration)}s)" if duration is not None else "## Motion beats"
    )
    sections.append(
        beats_header + "\n"
        + _motion_beats(
            action,
            camera,
            duration,
            effective_movement,
            visual_beats=shot.get("visual_beats"),
            static=is_static,
        )
    )

    camera_path = {
        "start_frame": str(shot.get("start_frame") or "").strip(),
        "end_frame": str(shot.get("end_frame") or "").strip(),
        "shot_size": str(shot.get("shot_size") or "").strip(),
        "camera_angle": str(shot.get("camera_angle") or "").strip(),
        "lens_intent": str(shot.get("lens_intent") or "").strip(),
    }
    sections.append(
        "## Camera\n"
        + _camera_direction(
            camera,
            effective_movement,
            camera_recipe,
            static=is_static,
            path=camera_path,
            subject=subject,
            location=locations[0] if locations else "",
            language=language,
        )
    )

    if prompt_skills.get("camera_movement"):
        sections.append("## Camera movement skill\n" + prompt_skills["camera_movement"])

    if prompt_skills.get("character_motion"):
        sections.append("## Character motion skill\n" + prompt_skills["character_motion"])

    movement_motivation = str(shot.get("movement_motivation") or "").strip()
    if movement_motivation:
        sections.append("## Movement motivation\n" + movement_motivation)

    sections.append("## Lighting & lens\n" + _lighting_and_lens(camera, style))

    blocking = _blocking_direction(composition, emotion, continuity_notes)
    if blocking:
        sections.append("## Blocking & visual intent\n" + blocking)

    if locations:
        sections.append(
            "## Location & environment\n"
            + _location_direction(locations, prompt_skills.get("location_identity", ""))
        )

    sections.append("## Foreground / background motion\n" + _layered_motion(subject))

    sections.append(
        "## Acting & emotion\n"
        + _acting_direction(characters, dialogue, prompt_skills.get("micro_expression", ""))
    )

    sections.append("## Sound\n" + _sound_direction(shot, language=language, duration=duration))

    continuity = _continuity_direction(context, caps, has_previous_shot)
    if continuity:
        sections.append("## Continuity\n" + continuity)

    sections.append("## Identity & references\n" + _reference_direction(shot, characters, caps))

    if knowledge_guidance.strip():
        # Retrieved recipe guidance can also carry [slot] placeholders; fill them here so no
        # raw bracket token leaks from any section (mirrors the camera-recipe weave above).
        guidance = fill_recipe_slots(
            knowledge_guidance.strip(),
            subject=subject,
            location=locations[0] if locations else "",
            language=language,
        )
        sections.append("## Retrieved filmmaking guidance\n" + guidance)

    style_prompt = format_style_prompt(style)
    if style_prompt:
        sections.append("## Style\n" + style_prompt)

    if has_style_reference:
        sections.append(STYLE_REFERENCE_GUARD)

    if prompt_skills.get("cinematography"):
        sections.append("## Cinematography skill\n" + prompt_skills["cinematography"])

    fmt_line = _format_direction(product_format)
    if fmt_line:
        sections.append("## Format\n" + fmt_line)

    avoid = list(_AVOID_TERMS) + [
        str(rule).strip() for rule in (hard_avoidances or []) if str(rule).strip()
    ]
    if dialogue:
        avoid.extend([
            "mumbling", "unintelligible speech", "gibberish", "made-up words",
            "wrong-language speech", "burned-in subtitles", "captions", "on-screen text",
            "hallucinated audio", "phantom laugh track",
        ])
    if is_static:
        tail = (
            ". The camera is intentionally still; do NOT invent camera motion. Avoid a frozen "
            "subject — the subject and scene must visibly move, act, and breathe every second."
        )
    else:
        tail = (
            ". Do not render a static or barely-moving image; every second must contain "
            "visible, intentional motion."
        )
    sections.append("## Avoid\n" + ", ".join(dict.fromkeys(avoid)) + tail)

    if duration is not None:
        sections.append(f"Duration ~{_fmt(duration)}s.")
    else:
        sections.append("Duration: use your default clip length.")
    return "\n\n".join(sections) + "\n"


def _sound_direction(
    shot: dict[str, Any], *, language: str = "en", duration: float | None = 2.0
) -> str:
    dialogue = [
        (
            str(item.get("character") or "Speaker").strip(),
            str(item.get("line") or "").strip(),
        )
        for item in (shot.get("dialogue") or [])
        if str(item.get("line") or "").strip()
    ]
    action = str(
        shot.get("action") or shot.get("description") or "the visible action"
    ).strip()
    locations = [
        str(value).strip()
        for value in (shot.get("reference_locations") or [])
        if str(value).strip()
    ]
    place = ", ".join(locations) or "the depicted location"

    narration = str(shot.get("narration") or "").strip()

    lines = []
    if dialogue:
        # Colon form (not quotes) + enunciation + pace-to-duration + no-subtitles is what makes
        # the model's native speech intelligible instead of gibberish.
        lines.append(
            "Spoken dialogue — generate exactly these lines, in order, clearly enunciated and "
            f"fully intelligible in {language}, with visible mouth movement matched to the words:"
        )
        lines.extend(f"- {speaker}: {line}" for speaker, line in dialogue)
        pace = (
            f"the shot's ~{_fmt(duration)}s" if duration is not None else "the shot's length"
        )
        lines.append(
            f"Speak at a natural pace that fits {pace}; do not rush. Do not "
            "invent, add, omit, paraphrase, reassign, or mumble any words, and produce no gibberish. "
            "No on-screen text, captions, or subtitles."
        )
    else:
        lines.append("Nobody speaks. No voices or intelligible words are heard.")
    if narration:
        # 旁白: non-diegetic narrator VO laid over the picture in the edit. The clip itself must NOT
        # lip-sync it — no on-screen character may mouth these words.
        lines.append(
            f"Off-screen narrator voiceover (added in the edit): {narration}. Do not lip-sync this "
            "narration to any character; no one on screen mouths or speaks these words."
        )
    lines.extend([
        f"Sound effects: synchronize concrete diegetic sounds to {action}.",
        f"Natural ambience: use believable room tone or environmental sound from {place}.",
        "No background music. No score, no singing, no beat, no melody, and no "
        "non-diegetic underscore.",
    ])
    return "\n".join(lines)


def _beat_size(beat: dict[str, Any]) -> float:
    """Relative timing size of a beat: its per-beat duration_s, else weight, else 1."""
    for key in ("duration_s", "weight"):
        value = beat.get(key)
        if value is None:
            continue
        try:
            size = float(value)
        except (TypeError, ValueError):
            continue
        if size > 0:
            return size
    return 1.0


def _motion_beats(
    action: str,
    camera: str,
    duration: float | None,
    camera_movement: str = "",
    *,
    visual_beats: list[dict[str, Any]] | None = None,
    static: bool = False,
) -> str:
    """Break the shot duration into ordered beats so motion evolves over time.

    Without a planned duration (model-default clip length), beats keep their order but
    carry no second marks — the model paces them across whatever length it renders.
    """
    ordered = [beat for beat in (visual_beats or []) if isinstance(beat, dict)]
    if ordered:
        sizes = [_beat_size(beat) for beat in ordered]
        explicit = any(
            beat.get("duration_s") is not None or beat.get("weight") is not None
            for beat in ordered
        )
        if duration is None:
            offsets = [0.0] * (len(ordered) + 1)
        elif not explicit:
            # Back-compat: no per-beat sizing → exact even split using the original
            # order of operations, byte-identical to the previous marker arithmetic.
            step = duration / len(ordered)
            offsets = [i * step for i in range(len(ordered) + 1)]
        else:
            total = sum(sizes) or 1.0
            offsets = [0.0]
            cum = 0.0
            for size in sizes:
                cum += size
                offsets.append(duration * cum / total)
        lines = []
        for i, beat in enumerate(ordered):
            marker = (
                f" ({_fmt(offsets[i])}–{_fmt(offsets[i + 1])}s)" if duration is not None else ""
            )
            detail = str(beat.get("action") or beat.get("description") or beat.get("id") or "").strip()
            secondary = str(beat.get("secondary_motion") or "").strip()
            sec_text = f" Secondary motion: {secondary.rstrip(' .')}." if secondary else ""
            states = beat.get("end_states") or {}
            state_text = ", ".join(f"{name}: {state}" for name, state in states.items())
            suffix = f" End at {state_text}." if state_text else ""
            lines.append(
                f"- Beat {i + 1}{marker}: {detail.rstrip(' .')}.{sec_text}{suffix}"
            )
        return "\n".join(lines)
    count = max(2, min(4, math.ceil(duration / 1.5))) if duration is not None else 3
    step = duration / count if duration is not None else None
    move = "held locked-off frame" if static else _camera_move_phrase(camera, camera_movement)
    action_text = (action or "the subject moves").rstrip(".")

    # Distinct progression words so each middle beat advances the action instead of
    # repeating one boilerplate string, and each ties the motion to the chosen camera move.
    middle_stages = ("takes hold", "builds", "intensifies", "carries through")
    beats = []
    for i in range(count):
        marker = (
            f" ({_fmt(i * step)}–{_fmt((i + 1) * step)}s)" if step is not None else ""
        )
        if i == 0:
            desc = f"establish the frame as {action_text[:1].lower() + action_text[1:]} begins"
        elif i == count - 1:
            desc = f"the motion resolves and settles as the {move} completes"
        else:
            stage = middle_stages[min(i - 1, len(middle_stages) - 1)]
            desc = (
                f"the action {stage}, carried by the {move} — visible physical change, "
                "weight, and follow-through"
            )
        beats.append(f"- Beat {i + 1}{marker}: {desc}")
    return "\n".join(beats)


def _state_trajectory(shot: dict[str, Any]) -> str:
    start = shot.get("start_states") or {}
    end = shot.get("end_states") or {}
    if not isinstance(start, dict) or not isinstance(end, dict) or not (start or end):
        return ""
    names = list(dict.fromkeys([*start.keys(), *end.keys()]))
    lines = [
        f"{name}: {start.get(name, 'unchanged')} → {end.get(name, start.get(name, 'unchanged'))}"
        for name in names
    ]
    lines.append(
        "Follow this order exactly. Preserve recognizable identity; never reveal a later state early."
    )
    return "\n".join(lines)


def _target_state_direction(shot: dict[str, Any], caps: VideoCapabilities) -> str:
    refs = [
        ref for ref in (shot.get("target_state_references") or [])
        if isinstance(ref, dict)
        and str(ref.get("character") or "").strip()
        and str(ref.get("state") or "").strip()
    ]
    if not refs:
        return ""
    selected_images = [
        str(path).replace("\\", "/").strip("/")
        for path in (shot.get("target_state_reference_images") or [])
        if str(path).strip()
    ]

    def has_supplied_media(ref: dict[str, Any]) -> bool:
        if not caps.supports_reference_images:
            return False
        image = str(ref.get("image") or "").replace("\\", "/").strip("/")
        return bool(image) and any(
            selected == image
            or selected.endswith("/" + image)
            or image.endswith("/" + selected)
            for selected in selected_images
        )

    supplied = [ref for ref in refs if has_supplied_media(ref)]
    text_only = [ref for ref in refs if ref not in supplied]
    parts: list[str] = []
    if supplied:
        labels = ", ".join(
            f"{str(ref['character']).strip()}: {str(ref['state']).strip()}"
            for ref in supplied
        )
        parts.append(
            "Final visible state must match the approved target-state reference exactly: "
            f"{labels}. Condition on the supplied target-state media for creature/body "
            "design, silhouette, palette, face, wardrobe remnants, and render style. "
            "Do not invent a different endpoint design during the transformation."
        )
    if text_only:
        labels = ", ".join(
            f"{str(ref['character']).strip()}: {str(ref['state']).strip()}"
            for ref in text_only
        )
        prefix = "For endpoints without supplied target-state media, " if supplied else ""
        parts.append(
            f"{prefix}final visual state lock: {labels}. Provider cannot receive extra "
            "target-state media for these endpoints in this request, so treat these state "
            "names as locked design constraints and carry the design from the starting "
            "keyframe plus the written state trajectory. Do not invent a different creature, "
            "costume, silhouette, palette, or render style."
        )
    return "\n".join(parts)


def _camera_direction(
    camera: str,
    camera_movement: str = "",
    camera_recipe: dict[str, Any] | None = None,
    *,
    static: bool = False,
    path: dict[str, str] | None = None,
    subject: str = "the subject",
    location: str = "",
    language: str = "en",
) -> str:
    base = _camera_base_direction(camera, camera_movement, static=static)
    path = path or {}
    path_lines = []
    if path.get("start_frame"):
        path_lines.append(f"Start framing: {path['start_frame']}.")
    if path.get("end_frame"):
        path_lines.append(f"End framing: {path['end_frame']}.")
    size_angle = ", ".join(p for p in (path.get("shot_size"), path.get("camera_angle")) if p)
    if size_angle:
        path_lines.append(f"Shot size / angle: {size_angle}.")
    if path.get("lens_intent"):
        path_lines.append(f"Lens intent: {path['lens_intent']}.")
    if path_lines and not static:
        path_lines.append("Carry the framing from start to end as ONE single move.")
    if path_lines:
        base = base + "\n" + " ".join(path_lines)
    recipe_text = str((camera_recipe or {}).get("prompt") or "").strip()
    if not recipe_text:
        return base
    recipe_text = fill_recipe_slots(
        recipe_text, subject=subject, location=location, language=language
    )
    name = str(camera_recipe.get("name") or camera_recipe.get("id") or "").strip()
    label = f" ({name})" if name else ""
    return (
        f"{base}\nCamera-movement recipe{label}: {recipe_text} "
        "This recipe is the precise execution of the camera movement above — perform it as "
        "the SAME single move (one move only), keeping the framing and subject above; do "
        "not add any second camera move."
    )


def _camera_base_direction(camera: str, camera_movement: str = "", *, static: bool = False) -> str:
    framing = camera or "balanced framing"
    if static:
        return (
            f"{framing}. Camera is locked-off and holds still on a tripod — the subject and "
            "environment carry the motion. Do not drift, push, or pan; this stillness is intentional."
        )
    if camera_movement:
        return f"{framing}. Camera movement: {camera_movement}. Sustain it with motivated timing."
    if not camera:
        return (
            "Camera holds balanced framing with a slow, subtle push-in so the frame is "
            "never locked off — one gentle motivated move, no jitter."
        )
    if _has_movement(camera):
        return f"{camera}. Sustain the move continuously across the shot."
    return (
        f"{camera}, with a {_camera_move_phrase(camera)} so the frame is never locked off."
    )


def _lighting_and_lens(camera: str, style: dict[str, Any]) -> str:
    low = camera.lower()
    if "close" in low:
        lens = "an 85mm lens with shallow depth of field"
    elif "wide" in low or "establish" in low or "long" in low:
        lens = "a 24-35mm wide lens holding deep focus"
    else:
        lens = "a 50mm lens at a natural perspective"
    palette = style.get("palette", "natural")
    return (
        f"Motivated, cinematic lighting that models form and holds mood ({palette}) as the "
        f"action moves; shoot on {lens}. Keep exposure and white balance stable across the shot."
    )


def _camera_move_phrase(camera: str, camera_movement: str = "") -> str:
    if camera_movement:
        return camera_movement
    low = camera.lower()
    if _has_movement(camera):
        return camera
    if "close" in low or low.strip() in {"cu", "ecu"}:
        return "slow push-in"
    if "wide" in low or "establish" in low or "long" in low:
        return "gentle drifting push-in"
    if "aerial" in low or "drone" in low:
        return "sweeping aerial move"
    # Neutral framing: default to a slow, motivated push-in rather than an unmotivated
    # handheld drift, which the camera-movement skill flags as jitter/warp-prone.
    return "slow, subtle push-in"


def _blocking_direction(composition: str, emotion: str, continuity_notes: str) -> str:
    lines = []
    if composition:
        lines.append(f"Composition: {composition}")
    if emotion:
        lines.append(f"Visible emotional intent: {emotion}")
    if continuity_notes:
        lines.append(f"Continuity notes: {continuity_notes}")
    return "\n".join(lines)


def _location_direction(locations: list[str], location_skill: str = "") -> str:
    text = (
        "Use the approved location bible for "
        + ", ".join(locations)
        + "; keep architecture, hero props, palette, materials, spatial layout, and light "
        "state consistent across the shot."
    )
    if location_skill:
        text += "\n" + location_skill
    return text


def _has_movement(camera: str) -> bool:
    low = camera.lower()
    return any(word in low for word in _MOVEMENT_WORDS)


def _layered_motion(subject: str) -> str:
    return (
        f"Foreground: {subject} and nearby elements move with intent and clear silhouette. "
        "Background: keep ambient life moving — drifting air, light, cloth, crowds, "
        "particles, or shifting shadows — so depth reads as alive, not a flat backdrop."
    )


def _acting_direction(characters: list[str], dialogue: list[str], micro_expression_skill: str = "") -> str:
    who = ", ".join(characters) if characters else "the subject"
    if dialogue:
        base = (
            f"{who} act the moment with readable emotion — eyes, breath, micro-expressions, "
            "and gestures timed to the dialogue."
        )
    else:
        base = (
        f"{who} convey emotion through body language, gaze, and posture; let feeling drive "
        "the movement rather than a held expression."
        )
    if micro_expression_skill:
        base += "\n" + micro_expression_skill
    return base


def _reference_direction(
    shot: dict[str, Any], characters: list[str], caps: VideoCapabilities
) -> str:
    seed = shot.get("reference_seed")
    has_refs = bool(shot.get("reference_images"))
    if caps.supports_reference_images:
        who = ", ".join(characters) if characters else "every named character"
        seed_txt = f" (locked reference seed {seed})" if seed is not None else ""
        ref_txt = (
            " Condition on the supplied bible reference images and turnaround/expression "
            "sheets so identity stays consistent."
            if has_refs
            else ""
        )
        return f"Hold the canonical look of {who}{seed_txt} from the show bible.{ref_txt}"
    return (
        "Keep each character's appearance locked to the starting keyframe; do not introduce "
        "new faces or wardrobe. No extra reference media is supplied for this provider."
    )


def _format_direction(product_format: dict[str, Any] | None) -> str:
    if not product_format:
        return ""
    parts: list[str] = []
    label = product_format.get("label") or product_format.get("name")
    if label:
        parts.append(str(label))
    pacing = product_format.get("pacing")
    if pacing:
        parts.append(f"pacing: {pacing}")
    structure = product_format.get("structure")
    if structure:
        parts.append(f"structure: {structure}")
    return " — ".join(parts)


def _story_beat(context: dict[str, Any]) -> str:
    parts: list[str] = []
    genre = str(context.get("genre") or "").strip()
    idea = context.get("idea")
    story = context.get("story") or {}
    scene = context.get("scene") or {}
    if genre:
        parts.append(f"Genre/题材 (lock the world and look to this): {genre}")
    if idea:
        parts.append(f"Project idea: {idea}")
    if story.get("logline"):
        parts.append(f"Story target: {story['logline']}")
    if story.get("synopsis"):
        parts.append(f"Story context: {story['synopsis']}")
    if story.get("themes"):
        parts.append("Themes: " + ", ".join(str(t) for t in story["themes"]))
    if scene.get("heading"):
        parts.append(f"Scene: {scene['heading']}")
    if scene.get("beats"):
        parts.append("Scene beats: " + " / ".join(str(b) for b in scene["beats"]))
    return "\n".join(f"- {part}" for part in parts)


def _performance_objective(shot: dict[str, Any], context: dict[str, Any], subject: str) -> str:
    action = str(shot.get("action") or "").strip()
    description = str(shot.get("description") or "").strip()
    has_dialogue = any(
        str(item.get("line") or "").strip()
        for item in (shot.get("dialogue") or [])
        if isinstance(item, dict)
    )
    objective = action or description
    lines = []
    if objective:
        lines.append(
            f"Make this shot clearly sell: {objective}. Every motion choice should support that beat."
        )
    else:
        lines.append(f"Make {subject}'s movement serve the scene beat, not generic motion.")
    if has_dialogue:
        lines.append(
            "For speaking performance, use only the exact dialogue in the Sound section; "
            "time facial acting and gestures to those lines."
        )
    lines.append("Prioritize readable intent, motivated blocking, and a clean final pose for the edit.")
    return " ".join(lines)


def _continuity_direction(
    context: dict[str, Any], caps: VideoCapabilities, has_previous_shot: bool
) -> str:
    prev = context.get("previous_shot") or {}
    next_shot = context.get("next_shot") or {}
    lines: list[str] = []

    if has_previous_shot and caps.supports_last_frame:
        lines.append(
            "Begin on the previous final frame and continue its motion so the cut "
            "is seamless; keep lighting, wardrobe, and framing consistent across the edit."
        )
    elif prev or next_shot:
        lines.append(
            "Maintain scene continuity across adjacent beats: same wardrobe, props, lighting, "
            "screen direction, and emotional escalation."
        )

    if caps.supports_last_frame:
        prev_line = _shot_line("Previous shot", prev)
    else:
        prev_line = _shot_line("Earlier beat", prev)
    next_line = _shot_line("Next shot", next_shot)
    lines.extend(line for line in (prev_line, next_line) if line)
    if next_shot:
        lines.append("End in a pose and camera direction that can cut naturally into the next beat.")
    return "\n".join(lines)


def _shot_line(label: str, shot: dict[str, Any] | None) -> str:
    if not shot:
        return ""
    action = shot.get("action") or shot.get("description") or ""
    camera = shot.get("camera") or ""
    sid = shot.get("id") or "adjacent"
    detail = " ".join(str(part) for part in (camera, action) if part).strip()
    return f"{label} ({sid}): {detail}" if detail else f"{label} ({sid})"


def _duration(value: Any) -> float | None:
    """The shot's planned seconds, or ``None`` when the video model's default governs."""
    if value is None:
        return None
    try:
        duration = float(value)
    except (TypeError, ValueError):
        duration = 2.0
    return duration if duration > 0 else 2.0


def _fmt(value: float) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.1f}"
