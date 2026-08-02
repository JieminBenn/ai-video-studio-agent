"""Insert one user-described shot into an existing shot list (clip or story mode).

The user says what the new shot should show and where it goes; one LLM call drafts a
full shot object consistent with its neighbors (same location, wardrobe, lighting,
visual state), and only the new shot then flows through the normal
keyframe → video-prompt → video stages. Existing shots' paid artifacts are never
touched (invariant #5); only whole-film aggregates (clips.json, audio.json, timeline,
output) are archived so the re-pended stages rebuild them to include the new shot.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .asset_regeneration import (
    _pending_stages,
    _rechain_shots,
    project_video_capabilities,
)
from .cinematography_recipes import load_camera_recipes, recipe_menu, resolve_recipe
from .clip_sequence import clamp_clip_duration, resolve_clip_plan
from .invalidation import archive_paths
from .language import language_label
from .orchestrator.project import Project
from .prompt_approvals import rebase_prompt_approvals
from .shot_design import normalize_shot_design
from .stages.base import Providers

_SHOT_ID_RE = re.compile(r"^sh-(\d+)$")
_HISTORY_ID_RE = re.compile(r"^sh-(\d+)\.")

# Aggregates rebuilt from the live shot list; archiving them (reversibly) makes the
# re-pended stages regenerate them to include the new shot without touching any
# existing per-shot artifact.
_AGGREGATE_PATHS = (
    "assets/clips/clips.json",
    "assets/qc/summary.json",
    "assets/audio/audio.json",
    "edit/timeline.json",
)

_INSERT_PROMPT_TEMPLATE = """[task:shot_insert]
You are adding ONE new shot into an existing, already-approved shot list. Design the
shot the user asked for so it fits seamlessly between its neighbors: keep the same
location, subject identity, wardrobe, lighting, palette, and visual/transformation
state as where the previous shot ends, flowing naturally into where the next shot
begins. Do not restage, relocate, or contradict either neighbor.

Return only valid JSON: one object with keys description, action, camera,
camera_movement, camera_recipe (an id from CAMERA RECIPES), composition, emotion,
dramatic_purpose, start_frame, end_frame, subject_blocking, shot_size, camera_angle,
lens_intent, movement_motivation, lighting_state, edit_relationship, duration_s
(realistic seconds for the shot), characters (names drawn from CAST), dialogue (list
of {{character, line}}; [] unless the request asks for speech), narration ("" unless
the request asks for an off-screen voiceover).

USER REQUEST (write the shot in this language, preserving requested words exactly):
{description}
LANGUAGE: {language}
CAST: {cast}
PREVIOUS SHOT: {previous_shot}
NEXT SHOT: {next_shot}
CAMERA MOVEMENT RULE: choose ONE motivated camera move at most; prefer the smallest
move, and default to a static/locked-off camera when the subject carries the motion.
CAMERA RECIPES:
{camera_recipe_menu}
"""

# Neighbor context the LLM needs for continuity — trimmed so the prompt stays small.
_NEIGHBOR_FIELDS = (
    "id", "scene", "description", "action", "camera", "camera_movement",
    "start_frame", "end_frame", "shot_size", "lighting_state", "emotion",
    "characters", "start_states", "end_states", "narration",
)


@dataclass
class AddShotResult:
    shot_id: str
    position: int
    planning_stage: str
    invalidated: list[str]
    archive_dir: str = ""


def next_shot_id(project: Project, shots: list[dict]) -> str:
    """Allocate a fresh ``sh-NNN`` id.

    Scans the live shot list AND the ``history/`` archive so a deleted shot's id is
    never reused — reuse would make restoring an archived family ambiguous. Order is
    the shot's position in shots.json, never its id.
    """
    highest = 0
    for shot in shots:
        match = _SHOT_ID_RE.match(str(shot.get("id") or ""))
        if match:
            highest = max(highest, int(match.group(1)))
    history = project.path("history")
    if history.is_dir():
        for path in history.rglob("sh-*"):
            match = _HISTORY_ID_RE.match(path.name) or _SHOT_ID_RE.match(path.stem)
            if match:
                highest = max(highest, int(match.group(1)))
    return f"sh-{highest + 1:03d}"


def add_shot(
    project: Project,
    *,
    description: str,
    after_shot_id: str = "",
    providers: Providers,
) -> AddShotResult:
    """Draft and insert one new shot after ``after_shot_id`` (blank = at the start)."""
    description = str(description or "").strip()
    if not description:
        raise ValueError("describe the new shot before adding it")
    if providers.llm is None:
        raise ValueError("this project profile does not include an LLM provider")
    shots_path = project.path("storyboard", "shots.json")
    if not shots_path.is_file():
        raise FileNotFoundError("storyboard/shots.json")
    shots = [
        s for s in json.loads(shots_path.read_text()).get("shots", [])
        if isinstance(s, dict)
    ]
    if not shots:
        raise ValueError("no shots to insert between — run the planning stage first")

    after = str(after_shot_id or "").strip()
    if after:
        if not any(s.get("id") == after for s in shots):
            raise FileNotFoundError(f"shot not found: {after}")
        position = next(i for i, s in enumerate(shots) if s.get("id") == after) + 1
    else:
        position = 0
    prev_shot = shots[position - 1] if position > 0 else None
    next_shot = shots[position] if position < len(shots) else None

    clip_mode = "clip" in project.stages
    planning_stage = "clip" if clip_mode else "storyboard"
    language = project.model_config.get("language") or "en"
    cast = _cast(project)
    recipes = load_camera_recipes()

    project.assert_budget_available()
    prompt = _INSERT_PROMPT_TEMPLATE.format(
        description=description,
        language=language_label(language),
        cast=", ".join(cast) or "the subject",
        previous_shot=_neighbor_view(prev_shot),
        next_shot=_neighbor_view(next_shot),
        camera_recipe_menu=recipe_menu(recipes, language=language) or "(none)",
    )
    gen = providers.llm.complete_json(prompt)
    draft = gen.content if isinstance(gen.content, dict) else {}
    project.add_generation_cost(stage=planning_stage, generation=gen)

    sid = next_shot_id(project, shots)
    shot = _build_shot(
        project,
        draft,
        sid,
        prev_shot=prev_shot,
        next_shot=next_shot,
        cast=cast,
        clip_mode=clip_mode,
        recipes=recipes,
        language=language,
    )
    shots.insert(position, shot)
    _rechain_shots(project, shots)
    shots_path.write_text(json.dumps({"shots": shots}, indent=2, ensure_ascii=False))

    if clip_mode:
        _insert_sequence_segment(project, position, shot)

    successor = shots[position + 1].get("id") if position + 1 < len(shots) else None
    invalidated_ids = [sid] + ([str(successor)] if successor else [])
    for kind in ("keyframes", "videos"):
        rebase_prompt_approvals(project, kind, invalidated_shot_ids=invalidated_ids)

    archive = archive_paths(
        project,
        [*_AGGREGATE_PATHS, f"output/{project.project_id}.mp4"],
        reason="shot-added",
        source_path="storyboard/shots.json",
        affected_shot_ids=[sid],
    )
    invalidated = _pending_stages(project, planning_stage)
    return AddShotResult(
        shot_id=sid,
        position=position,
        planning_stage=planning_stage,
        invalidated=invalidated,
        archive_dir=archive.archive_dir,
    )


def _build_shot(
    project: Project,
    draft: dict,
    sid: str,
    *,
    prev_shot: dict | None,
    next_shot: dict | None,
    cast: list[str],
    clip_mode: bool,
    recipes,
    language: str,
) -> dict:
    """Turn the LLM draft into a full normalized shot object (no references/prompts —
    the planning stage's idempotent backfill fills those on the dispatched re-run)."""
    from .stages.bible import seed_for
    from .stages.clip import _validated_clip_dialogue

    neighbor = prev_shot or next_shot or {}
    scene = 1 if clip_mode else neighbor.get("scene", 1)

    resolved = resolve_recipe(
        recipes, str(draft.get("camera_recipe") or "").strip(), language=language
    )
    characters = [
        name for name in (str(c).strip() for c in (draft.get("characters") or []))
        if name and (not cast or name in cast)
    ]
    if not characters:
        characters = list(neighbor.get("characters") or cast)

    shot = {
        "id": sid,
        "scene": scene,
        "description": str(draft.get("description") or "").strip(),
        "camera": str(draft.get("camera") or "").strip(),
        "camera_movement": str(draft.get("camera_movement") or "").strip(),
        "camera_recipe": resolved["id"] if resolved else "",
        "action": str(draft.get("action") or draft.get("description") or "").strip(),
        "composition": str(draft.get("composition") or "").strip(),
        "emotion": str(draft.get("emotion") or "").strip(),
        "continuity_notes": "",
        "dialogue": _validated_clip_dialogue(draft.get("dialogue"), cast),
        "duration_s": _shot_duration(project, draft, clip_mode=clip_mode),
        "characters": characters,
        "deps": [],  # rebuilt by _rechain_shots
        "keyframe": f"{sid}.png",
        "dramatic_purpose": str(draft.get("dramatic_purpose") or "").strip(),
        "start_frame": str(draft.get("start_frame") or "").strip(),
        "end_frame": str(draft.get("end_frame") or "").strip(),
        "subject_blocking": str(draft.get("subject_blocking") or "").strip(),
        "shot_size": str(draft.get("shot_size") or "").strip(),
        "camera_angle": str(draft.get("camera_angle") or "").strip(),
        "lens_intent": str(draft.get("lens_intent") or "").strip(),
        "movement_motivation": str(draft.get("movement_motivation") or "").strip(),
        "lighting_state": str(draft.get("lighting_state") or "").strip(),
        "edit_relationship": str(draft.get("edit_relationship") or "").strip(),
    }
    if not clip_mode:
        shot["narration"] = str(draft.get("narration") or "").strip()
    else:
        # One beat carrying the whole shot; states inherit the previous segment's end
        # state so the insertion never implies an unplanned transformation.
        carried = dict((prev_shot or {}).get("end_states") or {})
        shot["visual_beats"] = [{
            "id": f"{sid}-beat",
            "action": shot["action"] or shot["description"],
            "description": shot["description"],
            "camera": shot["camera"],
            "camera_movement": shot["camera_movement"],
            "start_frame": shot["start_frame"],
            "end_frame": shot["end_frame"],
            "start_states": dict(carried),
            "end_states": dict(carried),
        }]
        shot["start_states"] = dict(carried)
        shot["end_states"] = dict(carried)
    shot = normalize_shot_design(shot, prior_shot=prev_shot)
    shot.setdefault("reference_seed", seed_for(sid))
    return shot


def _shot_duration(project: Project, draft: dict, *, clip_mode: bool):
    """The new shot's duration: clip mode honors the project's clip plan and provider
    band (``None`` = model-default length); story mode takes the LLM's realistic guess."""
    if not clip_mode:
        try:
            return max(0.5, float(draft.get("duration_s")))
        except (TypeError, ValueError):
            return 2.0
    plan = resolve_clip_plan(
        project.model_config, project.model_config.get("product_format") or {}
    )
    capabilities = project_video_capabilities(project)
    seconds = plan.seconds if plan.mode in ("default", "manual") else draft.get("duration_s")
    return clamp_clip_duration(
        seconds,
        min_s=capabilities.min_duration_s,
        max_s=capabilities.max_duration_s,
    )


def _insert_sequence_segment(project: Project, position: int, shot: dict) -> None:
    """Keep visual_sequence.json index-aligned with shots.json in clip mode."""
    path = project.path("storyboard", "visual_sequence.json")
    if not path.is_file():
        return
    sequence = json.loads(path.read_text())
    segments = list(sequence.get("segments") or [])
    segment = {
        "id": f"segment-{shot['id']}",
        "duration_s": shot.get("duration_s"),
        "visual_beats": list(shot.get("visual_beats") or []),
        "start_states": dict(shot.get("start_states") or {}),
        "end_states": dict(shot.get("end_states") or {}),
    }
    segments.insert(min(position, len(segments)), segment)
    sequence["segments"] = segments
    sequence["clip_durations"] = [entry.get("duration_s") for entry in segments]
    known = [float(value) for value in sequence["clip_durations"] if value is not None]
    sequence["target_duration_s"] = sum(known) if known else None
    sequence["beats"] = [
        beat for entry in segments for beat in (entry.get("visual_beats") or [])
    ]
    path.write_text(json.dumps(sequence, indent=2, ensure_ascii=False) + "\n")


def _cast(project: Project) -> list[str]:
    path = project.path("story", "plot.json")
    if not path.is_file():
        return []
    plot = json.loads(path.read_text())
    return [c.get("name") for c in plot.get("characters", []) if c.get("name")]


def _neighbor_view(shot: dict | None) -> str:
    if not shot:
        return "(none)"
    view = {
        key: shot[key]
        for key in _NEIGHBOR_FIELDS
        if shot.get(key) not in (None, "", [], {})
    }
    return json.dumps(view, ensure_ascii=False)
