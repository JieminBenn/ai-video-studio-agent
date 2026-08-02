"""Clip stage: one global visual sequence -> the fewest coherent long clips.

Replaces the storyboard for short-video (clip) mode. It plans the complete ordered visual
arc once, then partitions it into provider-safe long shots (plus dialogue only when asked),
each pinned to scene 1 so the bible's location references attach, then reuses the
StoryboardStage keyframe/reference machinery verbatim (subclassing) so reference
conditioning is identical to story mode (invariant #4).

Writes the same files the video stage reads::

    storyboard/shots.json          ordered shots (recipe_id, duration_s, reference_*)
    storyboard/visual_sequence.json global intent, ordered beats, and state trajectory
    storyboard/keyframes/<id>.png  one reference-conditioned keyframe per shot

Idempotent (invariant #3): skips when complete; reuses an existing (hand-edited)
shots.json and only renders missing keyframes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import Providers, StageResult
from .bible import seed_for
from .storyboard import StoryboardStage, _shot_id
from ..cinematography_recipes import load_camera_recipes, recipe_menu, resolve_recipe
from ..runtime_skills import load_prompt_skill
from ..clip_sequence import clamp_clip_duration, normalize_visual_sequence, resolve_clip_plan
from ..language import language_label
from ..knowledge import packet_guidance
from ..providers.base import VideoCapabilities
from ..reference_assets import style_anchor_paths, style_reference_paths
from ..shot_design import normalize_shot_design

_PROMPT_TEMPLATE = """[task:clip]
You are a cinematographer designing ONE COMPLETE short-video visual sequence. It is not
a plot-heavy short film, but it must have a readable visual progression: hook, development,
and payoff. Plan the entire sequence once so identity, action, camera, and visual states stay
continuous across clip boundaries. Prefer a few long coherent clips over many tiny cuts.

Classify the idea. A transformation/reveal needs an ordered micro-arc; a pure showcase needs
motion, escalation, and a visual payoff. Include dialogue only when the IDEA explicitly asks
for speech or supplies spoken words; otherwise return no dialogue.

Return only valid JSON with keys:
- intent_class: transformation, reveal, showcase, performance, or micro-arc.
- complexity: simple, moderate, or complex, judged from the idea's actual content. A single
  continuous action or one quick reveal is simple and must read as ONE ~10-15s clip; a
  two-stage change is moderate; a multi-stage arc is complex. Be honest: most short prompts
  are simple. This caps total screen time (simple = one clip), so do not inflate it.
- recommended_duration_s: a soft fallback only. Prefer to express length through each
  beat's duration_s below; total screen time should reflect the idea's complexity.
- camera, camera_movement, camera_recipe, dramatic_purpose, start_frame, end_frame,
  subject_blocking, shot_size, camera_angle, lens_intent, movement_motivation,
  lighting_state, edit_relationship: global defaults for the sequence.
- craft_recipe_ids: relevant ids from RETRIEVED FILMMAKING KNOWLEDGE.
- description, composition, emotion.
- beats: 1-8 ordered objects, sized to complexity — use only 1-2 beats for a simple idea,
  more for a complex one. Beats are motion WITHIN clips, not separate clips; several short
  beats can share one clip. Every beat must contain id, action, secondary_motion (overlapping
  motion — cloth, hair, breath, weight shift), description, camera, camera_movement,
  start_frame, end_frame, duration_s (approximate seconds this beat needs on screen, sized
  to how much action it contains — a quick reveal ~4s, a sustained transformation ~10s), and
  start_states/end_states maps from character name to an exact state id from VISUAL STATE
  PLAN. Never show a later state in an earlier beat. Temporary effects belong only in the
  beats where they happen.
- composition: foreground/midground/background layout and focal point.
- emotion: the vibe the shot should project.
- dialogue: a list of objects with character and line. Preserve requested spoken words
  exactly; return [] unless the IDEA explicitly requests speech.

IDEA: {idea}
SUBJECTS: {subjects}
LANGUAGE: {language}
DURATION MODE: {duration_mode}
VISUAL STATE PLAN: {visual_state_plan}
CAMERA RECIPES:
{camera_recipe_menu}
CAMERA MOVEMENT RULE: choose ONE motivated camera move per beat; prefer the smallest move, and
default to a static/locked-off camera when the subject carries the motion. Never stack two moves.
Reserve [reserved] recipes (orbit, crane, aerial, whip) for beats whose drama truly earns them —
movement must be reasonable, never decorative.
CAMERA MOVEMENT SKILL (apply when choosing the move):
{camera_movement_skill}
CHARACTER MOTION SKILL (apply when writing each beat's action + secondary_motion):
{character_motion_skill}
RETRIEVED FILMMAKING KNOWLEDGE:
{knowledge}
"""

def _unique(values) -> list[str]:
    output: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in output:
            output.append(text)
    return output


def _selected_ids(packet: dict) -> list[str]:
    return [
        str(entry.get("id"))
        for entry in packet.get("selected_entries") or []
        if entry.get("id")
    ]


def _validated_clip_dialogue(raw_dialogue: Any, cast: list[str]) -> list[dict]:
    """Validate the LLM's structured ``dialogue`` list — language-agnostically.

    The LLM read the idea (in any language) and already decided whether there is
    speech and what is spoken; the prompt instructs it to return ``[]`` unless the idea
    requests speech and to preserve spoken words exactly. We do NOT re-interpret the idea
    text here (no English/Chinese word matching). We only apply structural validation: drop
    blank lines and coerce an off-cast speaker to the first cast member.
    """
    lines = []
    for item in list(raw_dialogue or []):
        if not isinstance(item, dict) or not str(item.get("line") or "").strip():
            continue
        speaker = str(item.get("character") or "").strip()
        if cast and speaker not in cast:
            speaker = cast[0]
        lines.append({
            "character": speaker or "Speaker",
            "line": str(item["line"]).strip(),
        })
    return lines


def _plan_prompt_mode(plan) -> str:
    """Describe the resolved clip plan for the LLM prompt (structured, not word-matched)."""
    if plan.mode == "default":
        return "fixed: exactly 1 clip; the video model decides the clip length"
    if plan.mode == "manual":
        seconds = f"{plan.seconds}s each" if plan.seconds else "length decided by the video model"
        return f"fixed: exactly {plan.count} clip(s), {seconds}"
    if plan.total_s:
        return f"total runtime {plan.total_s}s split into provider-legal clips"
    return "auto"


class ClipStage(StoryboardStage):
    name = "clip"

    _CONTINUITY_GUARD = (
        "Continue directly from the supplied previous frame: keep the same location, "
        "subject identity, wardrobe, lighting, palette, and transformation state as where "
        "the previous clip ended. A new camera angle, distance, or cut is allowed, but do "
        "NOT relocate, restage, swap the background, or jump to an unrelated scene."
    )

    def _chain_continuity(self, shots: list) -> None:
        """衔接: anchor each non-first clip's keyframe on the previous clip's keyframe so
        cuts continue (new angle allowed) instead of jumping. The previous keyframe is a
        subject reference (survives ``cap_reference_paths``) and the guard rides in the
        keyframe brief's Continuity section."""
        for prev, shot in zip(shots, shots[1:]):
            prev_kf = f"storyboard/keyframes/{prev['keyframe']}"
            refs = list(shot.get("keyframe_reference_images") or [])
            if prev_kf not in refs:
                refs.insert(0, prev_kf)
            shot["keyframe_reference_images"] = refs
            shot["continuity_reference"] = prev_kf
            guard = self._CONTINUITY_GUARD
            end_frame = str(prev.get("end_frame") or "").strip()
            if end_frame:
                guard += f" The previous clip ended on: {end_frame}."
            shot["continuity_notes"] = guard

    def run(self, project, providers: Providers) -> StageResult:
        if project.stage_status(self.name) == "complete":
            return StageResult(status="skipped", message="clip already complete")

        bible = self._load_bible(project)
        locations = self._load_locations(project)
        style_refs = style_anchor_paths(project)
        shots_path = project.path("storyboard", "shots.json")

        if shots_path.is_file():
            shots = json.loads(shots_path.read_text())["shots"]
            previous = None
            for shot in shots:
                self._fill_shot_references(
                    shot,
                    bible,
                    locations,
                    style_refs=style_refs,
                    blocked_style_refs=style_reference_paths(project),
                )
                shot.update(normalize_shot_design(shot, prior_shot=previous))
                packet = self._shot_packet(project, shot)
                shot["craft_recipe_ids"] = _unique(
                    list(shot.get("craft_recipe_ids") or [])
                    + _selected_ids(packet)
                )
                previous = shot
        else:
            shots = self._plan_clip_shots(project, bible, locations, providers, style_refs)
            shots_path.write_text(json.dumps({"shots": shots}, indent=2, ensure_ascii=False))

        self._chain_continuity(shots)
        self._fill_missing_keyframe_prompts(project, shots, bible, locations=locations)
        if self._grid_active(project, providers):
            for shot in shots:
                self._prepare_grid_prompt(project, shot, providers)
        else:
            for shot in shots:
                self._ensure_prompt_file(project, shot, providers)
        shots_path.write_text(json.dumps({"shots": shots}, indent=2, ensure_ascii=False))

        return StageResult(status="complete", message=f"clip: {len(shots)} shot(s)")

    def _plan_clip_shots(self, project, bible, locations, providers, style_refs):
        plot = json.loads(project.path("story", "plot.json").read_text())
        idea = project.story_dir.joinpath("idea.md").read_text().strip()
        language = project.model_config.get("language") or "en"
        fmt = project.model_config.get("product_format") or {}
        plan = resolve_clip_plan(project.model_config, fmt)
        cast = [c.get("name") for c in plot.get("characters", []) if c.get("name")]
        recipes = load_camera_recipes()
        menu = recipe_menu(recipes, language=language) or "(none)"

        states_path = project.path("story", "visual_state_changes.json")
        state_plan = json.loads(states_path.read_text()) if states_path.is_file() else {
            "version": 1, "characters": []
        }
        scene_packet = self._scene_packet(project, {
            "scene": 1,
            "beats": [idea],
            "emotional_turn": "deliver one decisive visual payoff",
        })
        prompt = _PROMPT_TEMPLATE.format(
            idea=idea,
            subjects=", ".join(cast) or "the subject",
            language=language_label(language),
            duration_mode=_plan_prompt_mode(plan),
            visual_state_plan=json.dumps(state_plan, ensure_ascii=False),
            camera_recipe_menu=menu,
            camera_movement_skill=load_prompt_skill("camera_movement") or "(none)",
            character_motion_skill=load_prompt_skill("character_motion") or "(none)",
            knowledge=packet_guidance(scene_packet),
        )
        project.assert_budget_available()
        gen = providers.llm.complete_json(prompt)
        raw = gen.content or {}
        project.add_generation_cost(stage=self.name, generation=gen)
        capabilities = getattr(providers.video, "capabilities", VideoCapabilities())
        # Tolerant parse: a hand-edited/non-numeric clip_max_total_s falls back to the
        # 60s default instead of crashing the stage.
        try:
            ceiling_s = max(
                1,
                int(round(float(
                    project.model_config.get("clip_max_total_s")
                    or fmt.get("clip_max_total_s")
                    or 60
                ))),
            )
        except (TypeError, ValueError):
            ceiling_s = 60
        sequence = normalize_visual_sequence(
            raw,
            plan=plan,
            max_clip_s=getattr(capabilities, "max_duration_s", 15),
            min_clip_s=getattr(capabilities, "min_duration_s", 4),
            ceiling_s=ceiling_s,
        )
        sequence_path = project.path("storyboard", "visual_sequence.json")
        sequence_path.write_text(json.dumps(sequence, indent=2, ensure_ascii=False) + "\n")

        shots = []
        prev_id = None
        for i, segment in enumerate(sequence["segments"], start=1):
            sid = _shot_id(i)
            beats = segment["visual_beats"]
            first = beats[0]
            last = beats[-1]
            recipe_id = first.get("camera_recipe") or raw.get("camera_recipe") or ""
            resolved = resolve_recipe(recipes, str(recipe_id).strip(), language=language)
            beat_actions = [str(beat.get("action") or "").strip() for beat in beats]
            beat_actions = [action for action in beat_actions if action]
            description = first.get("description") or raw.get("description") or "; then ".join(beat_actions)
            shot = normalize_shot_design({
                "id": sid,
                "scene": 1,                       # matches concept's location scene
                "description": description,
                "camera": first.get("camera") or raw.get("camera", ""),
                "camera_movement": str(first.get("camera_movement") or raw.get("camera_movement") or "").strip(),
                "camera_recipe": resolved["id"] if resolved else "",
                "craft_recipe_ids": _selected_ids(scene_packet),
                "action": "; then ".join(beat_actions),
                "composition": first.get("composition") or raw.get("composition", ""),
                "emotion": last.get("emotion") or raw.get("emotion", ""),
                "continuity_notes": "",
                "dialogue": _validated_clip_dialogue(raw.get("dialogue"), cast),
                "duration_s": segment["duration_s"],
                "characters": cast,
                "deps": [prev_id] if prev_id else [],
                "keyframe": f"{sid}.png",
                "dramatic_purpose": first.get("dramatic_purpose") or raw.get("dramatic_purpose", ""),
                "start_frame": first.get("start_frame") or raw.get("start_frame", ""),
                "end_frame": last.get("end_frame") or raw.get("end_frame", ""),
                "subject_blocking": first.get("subject_blocking") or raw.get("subject_blocking", ""),
                "shot_size": first.get("shot_size") or raw.get("shot_size", ""),
                "camera_angle": first.get("camera_angle") or raw.get("camera_angle", ""),
                "lens_intent": first.get("lens_intent") or raw.get("lens_intent", ""),
                "movement_motivation": first.get("movement_motivation") or raw.get("movement_motivation", ""),
                "lighting_state": first.get("lighting_state") or raw.get("lighting_state", ""),
                "edit_relationship": last.get("edit_relationship") or raw.get("edit_relationship", ""),
                "visual_beats": beats,
                "start_states": segment["start_states"],
                "end_states": segment["end_states"],
            }, prior_shot=shots[-1] if shots else None)
            self._fill_shot_references(
                shot,
                bible,
                locations,
                style_refs=style_refs,
                blocked_style_refs=style_reference_paths(project),
            )
            shot.setdefault("reference_seed", seed_for(f"clip-{i}"))
            shot_packet = self._shot_packet(project, shot)
            shot["craft_recipe_ids"] = _unique(
                list(shot.get("craft_recipe_ids") or [])
                + _selected_ids(shot_packet)
            )
            shots.append(shot)
            prev_id = sid
        return shots
