"""Storyboard stage: script scenes -> shots + reference-conditioned keyframes.

Breaks each scene into shots (camera, action, dialogue, duration, deps) via the LLM,
then renders one keyframe per shot. Each keyframe is **reference-conditioned** on the
bible: the shot's named characters and locations each thread their single combined
``reference.png`` model sheet and locked bible seeds into the keyframe, so identity
propagates from the reference images into every shot (invariant #4). Shots where a
character speaks also prompt the model to match the expression poses captured in that
combined sheet.
Sequential ``deps`` seed the shot dependency graph and continuity between shots.

Writes::

    storyboard/shots.json              ordered shots with their generation metadata
    storyboard/keyframes/<id>.png      one keyframe per shot

Idempotent (invariants #3/#5): skips the whole stage if complete; otherwise reuses an
existing (possibly hand-edited) shots.json and only renders keyframes that are missing,
so a re-run never duplicates a paid generation and never clobbers human edits.
Old-shape shots.json files (with ``reference_character``/``reference_seed`` but no
``reference_images``) are tolerated and backfilled on the next run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .base import Providers, Stage, StagePreflight, StageResult
from .bible import char_slug, seed_for
from ..cinematography_recipes import load_camera_recipes, recipe_menu, resolve_recipe
from ..clip_sequence import plan_scene_segments
from ..creative_brief import load_creative_brief, resolved_value
from ..creative_decisions import decision_preflight
from ..formats import genre_persona, genre_prompt, project_format
from ..identity_board import coerce_aliases
from ..image_generation import generate_project_image
from ..language import language_label
from ..motion_grid_prompt import compile_grid_prompt, panel_beats, resolve_layout
from ..image_prompt_director import (
    direct_grid_prompt,
    direct_image_prompt,
    directed_prompt_text,
    director_enabled,
)
from ..keyframe_prompt import compile_keyframe_prompt
from ..knowledge import (
    KnowledgeQuery,
    get_or_create_packet,
    load_packaged_core,
    packet_guidance,
)
from ..prompt_context import project_prompt_context
from ..prompt_approvals import confirm_prompt_batch
from ..providers.base import Generation, ImageCapabilities, VideoCapabilities
from ..reference_assets import (
    cap_reference_paths,
    live_reference_paths_for_shot,
    reference_intent_prompt,
    reference_paths_for_target,
    style_anchor_paths,
    style_reference_paths,
)
from ..runtime_skills import load_prompt_skill, load_prompt_skills
from ..shot_design import normalize_shot_design
from ..style import style_medium_lead

_PROMPT_TEMPLATE = """[task:storyboard]
You are {genre_persona}a director and cinematographer. Break the single screenplay scene below into
a shot list for AI image/video generation. Return only valid JSON with key `shots`
(ordered list).

Each shot must include:
- scene: scene number from the screenplay.
- camera: framing/lens shorthand, e.g. "wide establishing shot", "low-angle close-up".
- camera_movement: ONE motivated camera move, or "locked-off" when the subject should carry the
  motion (dialogue and intimate beats usually should be static). Never stack two moves.
- camera_recipe: the id of the SINGLE camera-movement recipe whose intent best fits this beat.
  Prefer the smallest motivated move; default to a calm/static recipe. Pick a [reserved] recipe
  (orbit, crane, aerial, whip) ONLY when the beat's drama truly earns it. Use "" for a plain locked-off
  hold. Movement must be reasonable — never decorative.
- dramatic_purpose: what changes for the audience during this shot.
- story_advance: the NEW story information this shot delivers versus every prior shot —
  a new action outcome, changed emotion, revelation, spatial/temporal move, or shifted
  relationship. Never restate a beat a neighboring shot already carries.
- start_frame and end_frame: concrete compositions before and after the decisive beat.
- subject_blocking: physical positions and movement through the frame.
- shot_size, camera_angle, lens_intent: precise visual perspective choices.
- movement_motivation: why camera movement or stillness serves the visible beat.
- lighting_state: source, direction, contrast, color, and continuity state.
- edit_relationship: the visual/emotional handoff to adjacent shots.
- craft_recipe_ids: relevant ids from RETRIEVED FILMMAKING KNOWLEDGE.
- action: concrete physical action that changes during the shot.
- visual_beats: a time-sequenced list of the character's concrete physical movements across the
  shot. Each beat is an object {{action, secondary_motion, weight}}: action = one clear body
  movement; secondary_motion = overlapping motion (cloth, hair, breath, weight shift); weight =
  relative on-screen share (1–3). Use 1–3 beats. If the action needs more distinct movements than
  fit cleanly, plan MORE shots rather than cramming beats.
- description: imageable one-sentence frame description with subject, setting, props,
  and visual effect.
- composition: foreground/midground/background layout and focal point.
- emotion: playable emotional intent or reaction visible in face/body.
- continuity_notes: screen direction, props, lighting, prior/next shot connection.
- dialogue: list of {{character, line}} for lines spoken in this shot.
- narration: off-screen narrator voiceover (旁白) carried by this shot, or "" — set it only
  when this shot voices the scene's narration; it is laid over the picture, never lip-synced.
- duration_s: realistic shot duration.
- characters: names appearing in the shot.

Quality bar:
- Every action must be visible motion, not an abstract story label.
- Use camera movement and composition to support the emotional turn.
- Plan enough shots to show setup, escalation, choice, and consequence for each scene.
- Avoid generic beats such as "Establish: ..." or "a small complication raises stakes"
  unless the shot says exactly what happens physically on screen.
- If a shot's story_advance duplicates a neighbor's, merge or cut it; adjacent shots must
  not repeat the same beat.
- Keep output compact: no markdown, no prose, no multiline string values, and keep
  each string value under 180 characters where possible.
- Plan only the provided scene; do not include shots for scenes not present in SCRIPT.

CAMERA RECIPES (pick camera_recipe by id; intents describe when each move lands):
{camera_recipe_menu}

CAMERA MOVEMENT SKILL (apply when choosing the move):
{camera_movement_skill}

CHARACTER MOTION SKILL (apply when writing visual_beats):
{character_motion_skill}

RETRIEVED FILMMAKING KNOWLEDGE (use only when it serves this scene):
{knowledge}

STORY FLOW SKILL (apply so each shot advances the story):
{story_flow_skill}

PRIOR SHOTS (the tail of what already happened on screen — do not repeat these beats):
{prior_shots}

{genre}{reference_intent}
SCRIPT: {script}
"""


_SCENE_SEQUENCE_TEMPLATE = """[task:scene_sequence]
You are a cinematographer planning ONE SCENE as a short sequence of coherent video clips for
a 分镜图 (storyboard-grid) workflow. Plan the fewest clips that tell the scene clearly: a
small scene is one short clip; a richer scene is a few longer clips. Lip-sync degrades across
long multi-shot clips, so keep every beat that has spoken dialogue as its OWN short beat (one
speaker, one line); merge only narration-only or action-only beats into longer clips.
Narration (旁白) is off-screen voiceover and may ride over a long clip.

Return only valid JSON with keys:
- intent_class: transformation, reveal, showcase, performance, or micro-arc.
- recommended_duration_s: total seconds the scene needs.
- camera, camera_movement, camera_recipe, dramatic_purpose, start_frame, end_frame,
  subject_blocking, shot_size, camera_angle, lens_intent, movement_motivation,
  lighting_state, edit_relationship, description, composition, emotion: scene defaults.
- beats: ordered objects, each with id, action, description, camera, camera_movement,
  start_frame, end_frame, start_states/end_states (character name -> exact state id),
  duration_s, dialogue (list of {{character, line}}; [] if none), narration ("" if none),
  and story_advance (the NEW story information this beat adds versus the previous beat;
  never restate a beat a neighbor already carries).
  A beat with dialogue must contain exactly one speaker's line and stay short.

LANGUAGE: {language}
GENRE PERSONA: {genre_persona}
CAMERA RECIPES: {camera_recipe_menu}
CAMERA MOVEMENT RULE: one motivated move per beat; default to the smallest move or a static
locked-off camera when the subject carries the motion; reserve [reserved] recipes (orbit, crane,
aerial, whip) for beats whose drama earns them. Never stack two moves.
CAMERA MOVEMENT SKILL (apply when choosing the move): {camera_movement_skill}
RETRIEVED FILMMAKING KNOWLEDGE: {knowledge}
STORY FLOW SKILL: {story_flow_skill}
{genre}{reference_intent}SCENE SCRIPT: {script}
"""


_FLOW_REVIEW_TEMPLATE = """[task:flow_review]
You are a film editor auditing a finished shot list for story flow (连贯性). Read the ordered
shots and find where consecutive shots repeat the same story beat — where the audience learns
nothing new from one shot to the next. Judge story information, not camera framing.

{story_flow_skill}

Return only valid JSON with keys:
- redundancies: list of {{shots: [shot ids], reason, suggestion: "merge"|"cut"|"differentiate"}}.
- progression: list of {{scene, advances (bool), note}} — does each scene keep advancing?
- summary: one line on overall flow health and the biggest issue.

SHOTS: {shots}
"""


def _shot_id(index: int) -> str:
    return f"sh-{index:03d}"


def _render_flow_review_md(review: dict) -> str:
    """Render the shot-sequence flow review as a human-readable gate artifact (advise-only)."""
    lines = ["# 连贯性审查 / Shot flow review", ""]

    lines.append("## 冗余镜头 / Redundant beats")
    redundancies = review.get("redundancies") or []
    if redundancies:
        for item in redundancies:
            shots = ", ".join(str(s) for s in (item.get("shots") or []))
            reason = str(item.get("reason") or "").strip()
            suggestion = str(item.get("suggestion") or "").strip()
            lines.append(f"- ⚠️ {shots}: {reason} → suggestion: {suggestion}")
    else:
        lines.append("- ✅ no redundant beats found")

    lines.append("")
    lines.append("## 推进 / Progression")
    progression = review.get("progression") or []
    if progression:
        for item in progression:
            scene = item.get("scene")
            advances = item.get("advances")
            note = str(item.get("note") or "").strip()
            mark = "✅" if advances else "⚠️"
            lines.append(f"- {mark} scene {scene}: {note}")
    else:
        lines.append("- (no per-scene notes)")

    summary = str(review.get("summary") or "").strip()
    if summary:
        lines.append("")
        lines.append(f"**Summary / 结论:** {summary}")

    return "\n".join(lines) + "\n"


class StoryboardStage(Stage):
    name = "storyboard"

    def preflight(
        self,
        project,
        providers: Providers,
        *,
        auto: bool = False,
    ) -> StagePreflight:
        brief = load_creative_brief(project)
        if resolved_value(brief, "camera_language"):
            return StagePreflight()
        return decision_preflight(
            project,
            providers,
            stage=self.name,
            gap={
                "dimension": "camera_language",
                "default": "patient and intimate",
                "why": "Camera language shapes shot distance, energy, and movement.",
            },
            auto=auto,
        )

    def run(self, project, providers: Providers) -> StageResult:
        if project.stage_status(self.name) == "complete":
            return StageResult(status="skipped", message="storyboard already complete")

        bible = self._load_bible(project)
        locations = self._load_locations(project)
        style_refs = style_anchor_paths(project)
        shots_path = project.path("storyboard", "shots.json")

        if shots_path.is_file():
            shots = json.loads(shots_path.read_text())["shots"]
            previous = None
            for shot in shots:                       # tolerate/backfill old-shape shots
                self._fill_shot_references(
                    shot,
                    bible,
                    locations,
                    style_refs=style_refs,
                    blocked_style_refs=style_reference_paths(project),
                )
                shot.update(normalize_shot_design(shot, prior_shot=previous))
                packet = self._shot_packet(project, shot)
                shot["craft_recipe_ids"] = _dedupe(
                    list(shot.get("craft_recipe_ids") or []) + _packet_ids(packet)
                )
                previous = shot
        elif self._story_grid_enabled(project, providers):
            shots = self._plan_grid_scene_shots(
                project, bible, locations, providers, style_refs=style_refs
            )
            shots_path.write_text(json.dumps({"shots": shots}, indent=2))
        else:
            shots = self._plan_shots(project, bible, locations, providers, style_refs=style_refs)
            shots_path.write_text(json.dumps({"shots": shots}, indent=2))

        self._fill_missing_keyframe_prompts(project, shots, bible, locations=locations)
        if self._story_grid_enabled(project, providers):
            for shot in shots:
                self._prepare_grid_prompt(project, shot, providers)
        else:
            for shot in shots:
                self._ensure_prompt_file(project, shot, providers)
        # Persist prompt/grid metadata and any backfilled shot fields.
        shots_path.write_text(json.dumps({"shots": shots}, indent=2))

        self._ensure_flow_review(project, shots, providers)

        return StageResult(status="complete", message=f"storyboard: {len(shots)} shot(s)")

    def on_approve(self, project, *, auto: bool = False) -> None:
        confirm_prompt_batch(
            project,
            "keyframes",
            confirmer="auto" if auto else "human",
        )

    def _load_bible(self, project) -> dict:
        """Map character name -> {seed, reference_images (project-relative), aliases}."""
        bible = {}
        chars_dir = project.path("bible", "characters")
        if not chars_dir.is_dir():
            return bible
        for cdir in chars_dir.iterdir():
            cfile = cdir / "character.json"
            if not cfile.is_file():
                continue
            data = json.loads(cfile.read_text())
            rel = []
            if (cdir / "reference.png").is_file():
                rel.append(f"bible/characters/{cdir.name}/reference.png")
            rel.extend(reference_paths_for_target(project, "character", data["name"], cdir.name))
            aliases = []
            appearance_lock = ""
            board_rules: dict[str, list[str]] = {}
            board = cdir / "identity_board.json"
            if board.is_file():
                board_data = json.loads(board.read_text())
                aliases = coerce_aliases(board_data.get("prompt_aliases"))
                appearance_lock = str(board_data.get("appearance_lock") or "").strip()
                board_rules = {
                    key: [str(x).strip() for x in (board_data.get(key) or []) if str(x).strip()]
                    for key in ("do", "dont", "hero_props", "continuity_priority")
                }
            state_references: dict[str, str] = {}
            states_path = cdir / "states.json"
            if states_path.is_file():
                states = json.loads(states_path.read_text())
                for state in states.get("states") or []:
                    if not isinstance(state, dict):
                        continue
                    sid = str(state.get("id") or "").strip()
                    state_rel = str(state.get("reference_image") or "").strip()
                    if not sid or not state_rel or state_rel == "reference.png":
                        continue
                    state_path = cdir / state_rel
                    if state_path.is_file():
                        state_references[sid] = (
                            state_path.relative_to(project.dir).as_posix()
                        )
            entry = {
                "slug": cdir.name,
                "seed": data.get("seed"),
                "reference_images": rel, "aliases": aliases,
                "appearance_lock": appearance_lock,
                "state_references": state_references,
                **board_rules,
            }
            bible[data["name"]] = entry
            # Also index by the stable folder slug. The slug is derived from the name the
            # character was FIRST created with (the plot name) and never changes, whereas the
            # display `name` can drift when a bible-text revision renames the character. Shots
            # still carry the original plot name, so this lets _bible_entry re-join them by
            # slug instead of silently dropping the character's reference sheet.
            bible.setdefault(cdir.name, entry)
        return bible

    def _load_locations(self, project) -> dict:
        """Map location name -> {scene_numbers, reference_images, aliases}."""
        locations = {}
        locs_dir = project.path("bible", "locations")
        if not locs_dir.is_dir():
            return locations
        for ldir in locs_dir.iterdir():
            lfile = ldir / "location.json"
            if not lfile.is_file():
                continue
            data = json.loads(lfile.read_text())
            rel = []
            if (ldir / "reference.png").is_file():
                rel.append(f"bible/locations/{ldir.name}/reference.png")
            name = data.get("name") or ldir.name
            rel.extend(reference_paths_for_target(project, "location", name, ldir.name))
            aliases = coerce_aliases(data.get("prompt_aliases")) or [name]
            locations[name] = {
                "scene_numbers": data.get("scene_numbers", []),
                "reference_images": rel,
                "aliases": aliases,
            }
        return locations

    @staticmethod
    def _prior_shots_block(prior: list[dict]) -> str:
        """Render the last shots as neighbor context so the planner cannot re-establish."""
        tail = [s for s in (prior or []) if isinstance(s, dict)][-2:]
        if not tail:
            return "(none)"
        lines = []
        for shot in tail:
            advance = str(shot.get("story_advance") or shot.get("action") or "").strip()
            action = str(shot.get("action") or "").strip()
            lines.append(f"- {shot.get('id', '?')}: {action} | advance: {advance}")
        return "\n".join(lines)

    @staticmethod
    def _shot_references(characters, bible: dict, scene):
        """Return (reference_characters, reference_images[rel], reference_seed)."""
        ref_chars, ref_images = [], []
        for name in characters or []:
            entry = _bible_entry(bible, name)
            if entry is not None:
                ref_chars.append(name)
                ref_images.extend(entry["reference_images"])
        # Seed: first bible character's locked seed, else a stable scene-derived seed.
        first = _bible_entry(bible, ref_chars[0]) if ref_chars else None
        if first is not None and first.get("seed") is not None:
            ref_seed = int(first["seed"])
        else:
            ref_seed = seed_for(f"scene-{scene}")
        return ref_chars, ref_images, ref_seed

    @staticmethod
    def _character_locks(ref_chars, bible: dict) -> dict[str, str]:
        """Map a shot's characters to their verbatim appearance locks, keyed by primary alias.

        Deduped by alias so a two-shot scene doesn't repeat the same lock. Characters without
        a stored lock are omitted (the reference-image path still conditions them)."""
        locks: dict[str, str] = {}
        for name in ref_chars or []:
            entry = _bible_entry(bible, name)
            if entry is None:
                continue
            lock = str(entry.get("appearance_lock") or "").strip()
            if not lock:
                continue
            alias = (entry.get("aliases") or [name])[0] or name
            locks.setdefault(alias, lock)
        return locks

    @staticmethod
    def _character_rules(ref_chars, bible: dict) -> dict[str, dict[str, list[str]]]:
        """Map a shot's characters to their identity-board do/dont/hero-prop/priority rules.

        Keyed by primary alias, deduped. Characters without any rule are omitted so the
        keyframe brief only carries a Character rules section when the bible defines one."""
        rules: dict[str, dict[str, list[str]]] = {}
        for name in ref_chars or []:
            entry = _bible_entry(bible, name)
            if entry is None:
                continue
            rule = {
                key: [str(x).strip() for x in (entry.get(key) or []) if str(x).strip()]
                for key in ("do", "dont", "hero_props", "continuity_priority")
            }
            if not any(rule.values()):
                continue
            alias = (entry.get("aliases") or [name])[0] or name
            rules.setdefault(alias, rule)
        return rules

    @staticmethod
    def _shot_location_references(scene, locations: dict) -> tuple[list[str], list[str]]:
        ref_locations, ref_images = [], []
        for name, data in locations.items():
            scenes = data.get("scene_numbers") or []
            if scenes and scene not in scenes and str(scene) not in {str(s) for s in scenes}:
                continue
            ref_locations.append(name)
            ref_images.extend(data.get("reference_images", []))
        return ref_locations, ref_images

    def _fill_shot_references(
        self,
        shot: dict,
        bible: dict,
        locations: dict,
        *,
        style_refs: list[str] | None = None,
        blocked_style_refs: list[str] | None = None,
    ) -> None:
        rc, char_images, rs = self._shot_references(
            shot.get("characters", []), bible, shot.get("scene")
        )
        rl, loc_images = self._shot_location_references(shot.get("scene"), locations)
        existing = list(shot.get("reference_images") or [])
        style_images = list(style_refs or [])
        blocked_style_images = set(blocked_style_refs or [])

        def _states(name) -> dict:
            # Resolve by slug-aware join so a renamed character's state sheets still bind.
            return (_bible_entry(bible, name) or {}).get("state_references", {})

        all_state_images = {
            path
            for name in rc
            for path in _states(name).values()
        }
        baseline_existing = [
            path
            for path in existing
            if path not in all_state_images and path not in blocked_style_images
        ]
        base_images = _dedupe(baseline_existing + char_images + loc_images + style_images)
        state_ids = []
        for values in (shot.get("start_states"), shot.get("end_states")):
            if isinstance(values, dict):
                state_ids.extend(str(value) for value in values.values())
        for beat in shot.get("visual_beats") or []:
            if not isinstance(beat, dict):
                continue
            for field in ("start_states", "end_states"):
                values = beat.get(field)
                if isinstance(values, dict):
                    state_ids.extend(str(value) for value in values.values())
        state_images = _dedupe([
            _states(name)[state_id]
            for name in rc
            for state_id in state_ids
            if state_id in _states(name)
        ])
        start_states = shot.get("start_states") or {}
        opening_state_images = _dedupe([
            _states(name)[str(start_states[name])]
            for name in rc
            if name in start_states
            and str(start_states[name]) in _states(name)
        ])
        opening_state_ids = {
            str(name): str(state)
            for name, state in (start_states.items() if isinstance(start_states, dict) else [])
        }
        target_state_references: list[dict[str, str]] = []

        def add_target_state_refs(values) -> None:
            if not isinstance(values, dict):
                return
            for name, state in values.items():
                name_text = str(name)
                state_id = str(state)
                if name_text not in rc or opening_state_ids.get(name_text) == state_id:
                    continue
                image = _states(name_text).get(state_id)
                if image:
                    target_state_references.append({
                        "character": name_text,
                        "state": state_id,
                        "image": image,
                    })

        add_target_state_refs(shot.get("end_states"))
        for beat in shot.get("visual_beats") or []:
            if isinstance(beat, dict):
                add_target_state_refs(beat.get("end_states"))
        target_state_references = _dedupe_state_references(target_state_references)
        target_state_images = _dedupe([ref["image"] for ref in target_state_references])
        shot["reference_characters"] = rc
        shot["reference_locations"] = rl
        shot["reference_style_images"] = style_images
        shot["state_reference_images"] = state_images
        shot["target_state_references"] = target_state_references
        shot["target_state_reference_images"] = target_state_images
        shot["keyframe_reference_images"] = _dedupe(base_images + opening_state_images)
        shot["reference_images"] = _dedupe(base_images + state_images)
        shot.setdefault("reference_seed", rs)
        shot.setdefault("camera_recipe", "")

    def _plan_shots(
        self,
        project,
        bible: dict,
        locations: dict,
        providers: Providers,
        style_refs: list[str] | None = None,
    ) -> list[dict]:
        script = json.loads(project.path("story", "script.json").read_text())
        style = dict(project.model_config.get("style") or {})
        fmt = project_format(project)
        language = project.model_config.get("language") or "en"
        recipes = load_camera_recipes()
        shots = []
        prev_id = None
        for raw in self._plan_raw_scene_shots(project, script, providers):
            i = len(shots) + 1
            sid = _shot_id(i)
            ref_chars, ref_images, ref_seed = self._shot_references(
                raw.get("characters", []), bible, raw.get("scene")
            )
            ref_locations, loc_images = self._shot_location_references(raw.get("scene"), locations)
            # Validate the planner's recipe id against the library; the compiled video
            # prompt derives its movement language from the recipe at compile time, so we
            # never persist a seeded camera_movement that could go stale on a recipe swap.
            resolved = resolve_recipe(recipes, str(raw.get("camera_recipe") or "").strip(),
                                      language=language)
            recipe_id = resolved["id"] if resolved else ""
            shot = normalize_shot_design({
                "id": sid,
                "scene": raw.get("scene"),
                "description": raw.get("description", ""),
                "camera": raw.get("camera", ""),
                "camera_movement": str(raw.get("camera_movement") or "").strip(),
                "camera_recipe": recipe_id,
                "craft_recipe_ids": list(raw.get("craft_recipe_ids") or []),
                "action": raw.get("action", ""),
                "visual_beats": raw.get("visual_beats", []),
                "composition": raw.get("composition", ""),
                "emotion": raw.get("emotion", ""),
                "continuity_notes": raw.get("continuity_notes", ""),
                "dialogue": raw.get("dialogue", []),
                "narration": raw.get("narration", ""),
                "duration_s": raw.get("duration_s", 2.0),
                "characters": raw.get("characters", []),
                "deps": [prev_id] if prev_id else [],
                "reference_characters": ref_chars,
                "reference_locations": ref_locations,
                "reference_style_images": list(style_refs or []),
                "reference_images": _dedupe(ref_images + loc_images + list(style_refs or [])),
                "reference_seed": ref_seed,
                "keyframe": f"{sid}.png",
                "dramatic_purpose": raw.get("dramatic_purpose", ""),
                "story_advance": raw.get("story_advance", ""),
                "start_frame": raw.get("start_frame", ""),
                "end_frame": raw.get("end_frame", ""),
                "subject_blocking": raw.get("subject_blocking", ""),
                "shot_size": raw.get("shot_size", ""),
                "camera_angle": raw.get("camera_angle", ""),
                "lens_intent": raw.get("lens_intent", ""),
                "movement_motivation": raw.get("movement_motivation", ""),
                "lighting_state": raw.get("lighting_state", ""),
                "edit_relationship": raw.get("edit_relationship", ""),
            }, prior_shot=shots[-1] if shots else None)
            # Match the backfill path (top of run()) so state/target-state reference
            # fields are present from first construction, not only appended on re-run.
            self._fill_shot_references(
                shot,
                bible,
                locations,
                style_refs=style_refs,
                blocked_style_refs=style_reference_paths(project),
            )
            packet = self._shot_packet(project, shot)
            shot["craft_recipe_ids"] = _dedupe(
                list(shot.get("craft_recipe_ids") or []) + _packet_ids(packet)
            )
            shots.append(shot)
            prev_id = sid
        self._fill_missing_keyframe_prompts(
            project, shots, bible, locations=locations, style=style, product_format=fmt
        )
        return shots

    @staticmethod
    def _flow_review_hash(shots: list[dict]) -> str:
        payload = [
            {
                "id": s.get("id"),
                "action": s.get("action"),
                "description": s.get("description"),
                "story_advance": s.get("story_advance"),
            }
            for s in shots
        ]
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()

    def _flow_review_enabled(self, project) -> bool:
        cfg = (project.model_config.get("storyboard") or {}).get("flow_review") or {}
        return bool(cfg.get("enabled", True))

    def _ensure_flow_review(self, project, shots: list[dict], providers: Providers) -> None:
        """Advise-only flow review over the full shot list; never mutates shots.json."""
        if not self._flow_review_enabled(project) or not shots:
            return
        review_path = project.path("storyboard", "shots.review.md")
        current = self._flow_review_hash(shots)
        marker = f"<!-- flow-review-hash: {current} -->"
        if review_path.is_file() and review_path.read_text().startswith(marker):
            return  # idempotent: shots unchanged since last review
        story_flow_skill = load_prompt_skills(["story_flow"]).get("story_flow", "(none)")
        digest = [
            {
                "id": s.get("id"),
                "scene": s.get("scene"),
                "action": s.get("action"),
                "description": s.get("description"),
                "story_advance": s.get("story_advance"),
            }
            for s in shots
        ]
        prompt = _FLOW_REVIEW_TEMPLATE.format(
            story_flow_skill=story_flow_skill,
            shots=json.dumps(digest, ensure_ascii=False),
        )
        project.assert_budget_available()
        gen = providers.llm.complete_json(prompt)
        project.add_generation_cost(stage=self.name, generation=gen)
        review = gen.content if isinstance(gen.content, dict) else {}
        review_path.write_text(marker + "\n" + _render_flow_review_md(review))

    def _plan_raw_scene_shots(self, project, script: dict, providers: Providers) -> list[dict]:
        raw_shots: list[dict] = []
        reference_intent = reference_intent_prompt(project)
        genre = genre_prompt(project.model_config)
        language = project.model_config.get("language") or "en"
        camera_recipe_menu = recipe_menu(load_camera_recipes(), language=language) or "(none)"
        story_flow_skill = load_prompt_skills(["story_flow"]).get("story_flow", "(none)")
        camera_movement_skill = load_prompt_skill("camera_movement") or "(none)"
        character_motion_skill = load_prompt_skill("character_motion") or "(none)"
        for scene_script in self._scene_scripts(script):
            scene = scene_script["episodes"][0]["scenes"][0]
            packet = self._scene_packet(project, scene)
            prompt = _PROMPT_TEMPLATE.format(
                genre_persona=genre_persona(project.model_config),
                camera_recipe_menu=camera_recipe_menu,
                camera_movement_skill=camera_movement_skill,
                character_motion_skill=character_motion_skill,
                knowledge=packet_guidance(packet),
                story_flow_skill=story_flow_skill,
                prior_shots=self._prior_shots_block(raw_shots),
                genre=genre,
                reference_intent=reference_intent,
                script=json.dumps(scene_script),
            )
            project.assert_budget_available()
            gen = providers.llm.complete_json(prompt)
            project.add_generation_cost(stage=self.name, generation=gen)
            scene_ids = _packet_ids(packet)
            scene_raw = list(gen.content.get("shots", []))
            # 旁白: scene-level narration is an off-screen voiceover for the whole scene.
            # Carry it onto the first shot of the scene unless the planner already placed
            # per-shot narration, so exactly one shot voices it (audio lays it over the cut).
            scene_narration = str(scene.get("narration") or "").strip()
            if scene_narration and not any(
                str(raw.get("narration") or "").strip() for raw in scene_raw
            ):
                if scene_raw:
                    scene_raw[0]["narration"] = scene_narration
            for raw in scene_raw:
                raw["craft_recipe_ids"] = _dedupe(
                    list(raw.get("craft_recipe_ids") or []) + scene_ids
                )
                raw_shots.append(raw)
        return raw_shots

    def _knowledge_query(
        self,
        project,
        *,
        intent_text: str,
        intents: tuple[str, ...],
    ) -> KnowledgeQuery:
        config = project.model_config or {}
        style = dict(config.get("style") or {})
        brief = load_creative_brief(project)
        return KnowledgeQuery(
            stage="storyboard",
            domains=(
                "camera_movement",
                "lens_angle",
                "lighting",
                "composition",
                "performance",
                "editing",
            ),
            intent_text=" ".join(filter(None, [
                project.idea,
                resolved_value(brief, "tone"),
                resolved_value(brief, "camera_language"),
                intent_text,
            ])),
            intents=intents,
            style=str(config.get("style_name") or style.get("look") or ""),
            format_name=str(
                config.get("format_name")
                or (config.get("product_format") or {}).get("name")
                or ""
            ),
            language=str(config.get("language") or "en"),
            capabilities=frozenset({"camera_motion"}),
            hard_avoidances=tuple(brief.get("hard_avoidances") or []),
        )

    def _scene_packet(self, project, scene: dict) -> dict:
        query = self._knowledge_query(
            project,
            intent_text=json.dumps(scene, ensure_ascii=False, sort_keys=True),
            intents=("continuity", "pacing"),
        )
        retrieval = project.model_config.get("knowledge_retrieval") or {}
        return get_or_create_packet(
            project,
            purpose="scene",
            target=str(scene.get("scene") or "unknown"),
            query=query,
            entries=load_packaged_core(),
            limit=int(retrieval.get("max_entries", 6)),
        )

    def _shot_packet(self, project, shot: dict) -> dict:
        query = self._knowledge_query(
            project,
            intent_text=" ".join(str(shot.get(key) or "") for key in (
                "dramatic_purpose",
                "action",
                "emotion",
                "camera",
                "movement_motivation",
                "lighting_state",
            )),
            intents=("continuity", "pacing"),
        )
        retrieval = project.model_config.get("knowledge_retrieval") or {}
        return get_or_create_packet(
            project,
            purpose="shot",
            target=str(shot.get("id") or "unknown"),
            query=query,
            entries=load_packaged_core(),
            limit=int(retrieval.get("max_entries", 6)),
        )

    @staticmethod
    def _scene_scripts(script: dict) -> list[dict]:
        scene_scripts = []
        for episode in script.get("episodes", []):
            episode_no = episode.get("episode")
            for scene in episode.get("scenes", []):
                scene_scripts.append({
                    "title": script.get("title"),
                    "episodes": [{
                        "episode": episode_no,
                        "scenes": [scene],
                    }],
                })
        return scene_scripts

    @staticmethod
    def _keyframe_prompt(
        raw: dict,
        ref_chars,
        bible: dict,
        style: dict,
        product_format=None,
        context: dict | None = None,
        locations: dict | None = None,
        prompt_skills: dict[str, str] | None = None,
        knowledge_guidance: str = "",
        hard_avoidances: list[str] | None = None,
    ) -> str:
        # Resolve identity aliases + whether a featured speaking character has an
        # expression sheet, then hand off to the structured compiler (invariant #4/#8).
        ref_entries = [(c, _bible_entry(bible, c)) for c in ref_chars]
        aliases = [
            a for c, entry in ref_entries if entry is not None
            for a in (entry["aliases"] or [c])
        ]
        locations = locations or {}
        ref_locations = raw.get("reference_locations", [])
        location_aliases = [
            a for loc in ref_locations if loc in locations for a in (locations[loc]["aliases"] or [loc])
        ]
        has_expression_sheet = any(
            entry is not None and entry["reference_images"] for _c, entry in ref_entries
        )
        character_locks = StoryboardStage._character_locks(ref_chars, bible)
        character_rules = StoryboardStage._character_rules(ref_chars, bible)
        return compile_keyframe_prompt(
            raw,
            ref_aliases=aliases,
            location_aliases=location_aliases,
            has_expression_sheet=has_expression_sheet,
            speaking=bool(raw.get("dialogue")),
            style=style,
            product_format=product_format,
            context=context,
            prompt_skills=prompt_skills,
            knowledge_guidance=knowledge_guidance,
            hard_avoidances=hard_avoidances,
            character_locks=character_locks,
            character_rules=character_rules,
            has_style_reference=bool(raw.get("reference_style_images")),
        ).strip()

    def _fill_missing_keyframe_prompts(
        self,
        project,
        shots: list[dict],
        bible: dict,
        *,
        locations: dict | None = None,
        style: dict | None = None,
        product_format=None,
    ) -> None:
        style = dict(style if style is not None else project.model_config.get("style") or {})
        fmt = product_format if product_format is not None else project_format(project)
        locations = locations if locations is not None else self._load_locations(project)
        prompt_skills = load_prompt_skills([
            "character_identity",
            "location_identity",
        ])
        for shot in shots:
            packet = self._shot_packet(project, shot)
            knowledge_path = project.path(
                "storyboard", "prompts", f"{shot['id']}.knowledge.json"
            )
            if not knowledge_path.is_file():
                knowledge_path.parent.mkdir(parents=True, exist_ok=True)
                knowledge_path.write_text(
                    json.dumps(packet, ensure_ascii=False, indent=2) + "\n"
                )
            ref_chars = shot.get("reference_characters", [])
            locks = self._character_locks(ref_chars, bible)
            if locks:
                shot.setdefault("character_appearance_locks", list(locks.values()))
            rules = self._character_rules(ref_chars, bible)
            dont_rules = [d for rule in rules.values() for d in rule.get("dont", [])]
            if dont_rules:
                shot.setdefault("character_dont_rules", dont_rules)
            if shot.get("keyframe_prompt"):
                continue
            context = project_prompt_context(project, shot, shots=shots)
            shot["keyframe_prompt"] = self._keyframe_prompt(
                shot,
                ref_chars,
                bible,
                locations=locations,
                style=style,
                product_format=fmt,
                context=context,
                prompt_skills=prompt_skills,
                # Motion/camera recipes remain a saved artifact for video prompting,
                # but never cross into the provider-bound static-image prompt.
                knowledge_guidance="",
                hard_avoidances=list(
                    load_creative_brief(project).get("hard_avoidances") or []
                ),
            )

    @staticmethod
    def _prompt_path(project, shot: dict):
        return project.path("storyboard", "prompts", f"{shot['id']}.keyframe.md")

    @staticmethod
    def _brief_path(project, shot: dict):
        return project.path("storyboard", "prompts", f"{shot['id']}.brief.md")

    def _ensure_brief_file(self, project, shot: dict):
        """Persist the compiled structured brief (input to the director)."""
        brief_path = self._brief_path(project, shot)
        brief = shot.get("keyframe_prompt", "").strip()
        if not brief:
            brief = (
                f"{shot.get('camera', '')}. {shot.get('action', '')}. "
                f"{shot.get('description', '')}"
            ).strip()
            shot["keyframe_prompt"] = brief
        if not brief_path.is_file():
            brief_path.parent.mkdir(parents=True, exist_ok=True)
            brief_path.write_text(brief)
        return brief_path

    def _ensure_prompt_file(self, project, shot: dict, providers: Providers):
        """Return the hand-editable final prompt sent to the image model.

        The director (invariant #6) rewrites the brief into a dense, model-ready prompt.
        If the keyframe prompt already exists it is reused verbatim — never re-directed —
        so re-runs never duplicate the call or clobber human edits (invariants #3/#8).
        """
        prompt_path = self._prompt_path(project, shot)
        brief_path = self._ensure_brief_file(project, shot)
        if prompt_path.is_file():
            return prompt_path
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        final = self._direct_prompt(project, shot, providers, brief=brief_path.read_text())
        prompt_path.write_text(final)
        return prompt_path

    def _direct_generation(
        self,
        project,
        shot: dict,
        providers: Providers,
        *,
        brief: str,
    ) -> Generation:
        config = project.model_config or {}
        if not director_enabled(config) or providers.llm is None:
            return Generation(
                content={"prompt": brief, "negative": ""},
                provider="compiled",
                model="deterministic-compiler",
            )
        style = dict(config.get("style") or {})
        return direct_image_prompt(
            providers,
            project,
            stage=self.name,
            brief=brief,
            style=style,
            style_name=str(config.get("style_name") or style.get("look") or ""),
            intent=str(config.get("creative_intent") or project.idea or ""),
            purpose="keyframe",
            reference_aliases=list(shot.get("reference_characters") or []),
            verbatim_locks=list(shot.get("character_appearance_locks") or []),
            style_signature=style_medium_lead(style),
            max_prompt_chars=getattr(
                getattr(providers.image, "capabilities", None), "max_prompt_length", None
            ),
            hard_avoidances=(
                list(load_creative_brief(project).get("hard_avoidances") or [])
                + [str(r).strip() for r in (shot.get("character_dont_rules") or []) if str(r).strip()]
            ),
            language=str(config.get("language") or "en"),
        )

    def _direct_prompt(self, project, shot: dict, providers: Providers, *, brief: str) -> str:
        gen = self._direct_generation(project, shot, providers, brief=brief)
        final = directed_prompt_text(gen.content if isinstance(gen.content, dict) else {})
        return final or brief

    def _render_missing_keyframes(self, project, shots: list[dict], providers: Providers) -> None:
        for shot in shots:
            prompt_path = self._ensure_prompt_file(project, shot, providers)
            kf = project.path("storyboard", "keyframes", shot["keyframe"])
            if kf.is_file():
                continue
            reference_fields = (
                ("keyframe_reference_images",)
                if "keyframe_reference_images" in shot
                else ("reference_images",)
            )
            rel_refs = live_reference_paths_for_shot(
                project,
                shot,
                reference_fields,
            )
            abs_refs = [str(project.dir / ref) for ref in rel_refs]
            abs_refs = [r for r in abs_refs if Path(r).is_file()]
            named_rel = set(shot.get("named_reference_images") or [])
            named_refs = [
                str(project.dir / ref)
                for ref in rel_refs
                if ref in named_rel and str(project.dir / ref) in abs_refs
            ]
            abs_refs = cap_reference_paths(
                project,
                abs_refs,
                providers.image.capabilities.max_reference_images,
                named_refs=named_refs,
            )
            # Real consistency comes from reference_images; seed still drives deterministic
            # fake output and is a provider hint where supported.
            gen = generate_project_image(
                project,
                providers.image,
                prompt_path.read_text(), out_path=str(kf),
                reference_images=abs_refs, seed=shot["reference_seed"],
            )
            project.add_generation_cost(stage=self.name, generation=gen)

    # ------------------------------------------------------------------
    # Grid-aware keyframe rendering (分镜图)
    # ------------------------------------------------------------------
    def _grid_active(self, project, providers: Providers) -> bool:
        """Grid path on only when enabled AND both providers are grid-capable."""
        cfg = project.model_config.get("motion_grid") or {}
        if not bool(cfg.get("enabled", False)):
            return False
        img = getattr(providers.image, "capabilities", ImageCapabilities())
        vid = getattr(providers.video, "capabilities", VideoCapabilities())
        return bool(
            getattr(img, "supports_storyboard_grid", False)
            and getattr(vid, "supports_storyboard_grid", False)
        )

    def _story_grid_enabled(self, project, providers: Providers) -> bool:
        cfg = project.model_config.get("motion_grid") or {}
        return bool(cfg.get("story_scenes")) and self._grid_active(project, providers)

    @staticmethod
    def _segment_characters(segment: dict) -> list[str]:
        names: list[str] = []
        for beat in segment.get("visual_beats") or []:
            for name in beat.get("characters") or []:
                if name and name not in names:
                    names.append(name)
            for line in beat.get("dialogue") or []:
                speaker = str(line.get("character") or "").strip()
                if speaker and speaker not in names:
                    names.append(speaker)
        return names

    def _plan_grid_scene_shots(
        self, project, bible: dict, locations: dict, providers: Providers,
        style_refs: list[str] | None = None,
    ) -> list[dict]:
        script = json.loads(project.path("story", "script.json").read_text())
        language = project.model_config.get("language") or "en"
        recipes = load_camera_recipes()
        capabilities = getattr(providers.video, "capabilities", VideoCapabilities())
        max_clip_s = getattr(capabilities, "max_duration_s", 15)
        reference_intent = reference_intent_prompt(project)
        genre = genre_prompt(project.model_config)
        camera_recipe_menu = recipe_menu(recipes, language=language) or "(none)"
        story_flow_skill = load_prompt_skills(["story_flow"]).get("story_flow", "(none)")
        camera_movement_skill = load_prompt_skill("camera_movement") or "(none)"

        shots: list[dict] = []
        prev_id = None
        seq_dir = project.path("storyboard", "scene_sequences")
        seq_dir.mkdir(parents=True, exist_ok=True)
        for scene_script in self._scene_scripts(script):
            scene = scene_script["episodes"][0]["scenes"][0]
            scene_no = scene.get("scene")
            packet = self._scene_packet(project, scene)
            prompt = _SCENE_SEQUENCE_TEMPLATE.format(
                language=language_label(language),
                genre_persona=genre_persona(project.model_config),
                camera_recipe_menu=camera_recipe_menu,
                camera_movement_skill=camera_movement_skill,
                knowledge=packet_guidance(packet),
                story_flow_skill=story_flow_skill,
                genre=genre,
                reference_intent=reference_intent,
                script=json.dumps(scene_script, ensure_ascii=False),
            )
            project.assert_budget_available()
            gen = providers.llm.complete_json(prompt)
            raw = gen.content or {}
            project.add_generation_cost(stage=self.name, generation=gen)
            segments = plan_scene_segments(raw.get("beats"), max_clip_s=max_clip_s, min_clip_s=4)
            # 旁白: scene-level narration rides the first clip unless a beat already voices one.
            scene_narration = str(scene.get("narration") or "").strip()
            if scene_narration and segments and not any(s["narration"] for s in segments):
                segments[0]["narration"] = scene_narration
            (seq_dir / f"{scene_no}.json").write_text(
                json.dumps(
                    {"version": 1, "scene": scene_no, "segments": segments},
                    indent=2, ensure_ascii=False,
                ) + "\n"
            )
            scene_ids = _packet_ids(packet)
            for segment in segments:
                sid = _shot_id(len(shots) + 1)
                beats = segment["visual_beats"]
                first, last = beats[0], beats[-1]
                actions = [
                    str(b.get("action") or "").strip()
                    for b in beats if str(b.get("action") or "").strip()
                ]
                resolved = resolve_recipe(
                    recipes, str(first.get("camera_recipe") or raw.get("camera_recipe") or "").strip(),
                    language=language,
                )
                characters = self._segment_characters(segment)
                shot = normalize_shot_design({
                    "id": sid,
                    "scene": scene_no,
                    "description": first.get("description") or raw.get("description", "")
                        or "; then ".join(actions),
                    "camera": first.get("camera") or raw.get("camera", ""),
                    "camera_movement": str(
                        first.get("camera_movement") or raw.get("camera_movement") or ""
                    ).strip(),
                    "camera_recipe": resolved["id"] if resolved else "",
                    "craft_recipe_ids": list(scene_ids),
                    "action": "; then ".join(actions),
                    "composition": first.get("composition") or raw.get("composition", ""),
                    "emotion": last.get("emotion") or raw.get("emotion", ""),
                    "continuity_notes": "",
                    "dialogue": segment["dialogue"],
                    "narration": segment["narration"],
                    "duration_s": segment["duration_s"],
                    "characters": characters,
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
                self._fill_shot_references(shot, bible, locations, style_refs=style_refs)
                shot_packet = self._shot_packet(project, shot)
                shot["craft_recipe_ids"] = _dedupe(
                    list(shot.get("craft_recipe_ids") or []) + _packet_ids(shot_packet)
                )
                shots.append(shot)
                prev_id = sid
        return shots

    def _render_keyframes_or_grids(self, project, shots: list, providers: Providers) -> None:
        if not self._grid_active(project, providers):
            self._render_missing_keyframes(project, shots, providers)
            return
        for shot in shots:
            self._render_grid_keyframe(project, shot, providers)

    def _prepare_grid_prompt(self, project, shot: dict, providers: Providers) -> Path:
        """Persist one editable grid prompt without calling the image provider."""
        cfg = project.model_config.get("motion_grid") or {}
        rows, cols, n = resolve_layout(
            str(cfg.get("layout") or "auto"), duration_s=shot.get("duration_s", 4.0)
        )
        shot["motion_grid"] = {
            "layout": f"{rows}x{cols}", "rows": rows, "cols": cols,
            "panel_count": n,
            "panel_beats": panel_beats(
                shot.get("action", ""), n, visual_beats=shot.get("visual_beats")
            ),
        }

        style = dict(project.model_config.get("style") or {})
        brief = compile_grid_prompt(
            shot, rows=rows, cols=cols, style=style,
            product_format=project_format(project),
            context=project_prompt_context(project, shot, shots=[shot]),
            prompt_skills=load_prompt_skills(["character_identity", "location_identity"]),
            character_locks=list(shot.get("character_appearance_locks") or []),
        )
        brief_path = project.path("storyboard", "prompts", f"{shot['id']}.grid.brief.md")
        brief_path.parent.mkdir(parents=True, exist_ok=True)
        if not brief_path.is_file():
            brief_path.write_text(brief)
        prompt_path = project.path(
            "storyboard", "prompts", f"{shot['id']}.grid.md"
        )
        if not prompt_path.is_file():
            final = self._direct_grid(
                project,
                shot,
                providers,
                brief=brief_path.read_text(),
            )
            prompt_path.write_text(final)
        return prompt_path

    def _render_grid_keyframe(
        self,
        project,
        shot: dict,
        providers: Providers,
        *,
        prepared_only: bool = False,
    ) -> None:
        """Render an already-prepared 分镜图 prompt as the shot keyframe."""
        prompt_path = project.path(
            "storyboard", "prompts", f"{shot['id']}.grid.md"
        )
        if not prepared_only:
            prompt_path = self._prepare_grid_prompt(project, shot, providers)
        elif not prompt_path.is_file():
            raise FileNotFoundError(f"prepared grid prompt is missing: {prompt_path}")
        kf = project.path("storyboard", "keyframes", shot["keyframe"])
        if kf.is_file():
            return

        abs_refs = [
            str(project.dir / r)
            for r in shot.get("keyframe_reference_images", shot.get("reference_images", []))
        ]
        abs_refs = [r for r in abs_refs if Path(r).is_file()]
        abs_refs = cap_reference_paths(
            project, abs_refs, providers.image.capabilities.max_reference_images
        )
        gen = generate_project_image(
            project,
            providers.image,
            prompt_path.read_text(),
            out_path=str(kf),
            reference_images=abs_refs,
            seed=shot["reference_seed"],
        )
        project.add_generation_cost(stage=self.name, generation=gen)

    def _direct_grid(self, project, shot: dict, providers: Providers, *, brief: str) -> str:
        config = project.model_config or {}
        if not director_enabled(config) or providers.llm is None:
            return brief
        style = dict(config.get("style") or {})
        gen = direct_grid_prompt(
            providers, project, brief=brief, style=style,
            style_name=str(config.get("style_name") or style.get("look") or ""),
            intent=str(config.get("creative_intent") or project.idea or ""),
            reference_aliases=list(shot.get("reference_characters") or []),
            verbatim_locks=list(shot.get("character_appearance_locks") or []),
            style_signature=style_medium_lead(style),
            max_prompt_chars=getattr(
                getattr(providers.image, "capabilities", None), "max_prompt_length", None
            ),
            language=str(config.get("language") or "en"),
        )
        final = directed_prompt_text(gen.content if isinstance(gen.content, dict) else {})
        return final or brief


def _bible_entry(bible: dict, name) -> dict | None:
    """Resolve a shot's character name to its bible entry.

    Joins by exact display name first, then by the STABLE folder slug (``char_slug`` of the
    name the bible folder was created from). This survives a bible-text revision that renamed
    the character's display ``name`` — shots keep the original plot name, which would otherwise
    fail an exact-name join and silently unlink every shot from its reference sheet, so a
    regenerated character never reached downstream keyframes/clips (invariant #4).
    """
    if name is None:
        return None
    entry = bible.get(name)
    if entry is not None:
        return entry
    return bible.get(char_slug(str(name)))


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _dedupe_state_references(values: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = set()
    out: list[dict[str, str]] = []
    for value in values:
        key = (
            str(value.get("character") or ""),
            str(value.get("state") or ""),
            str(value.get("image") or ""),
        )
        if not all(key) or key in seen:
            continue
        seen.add(key)
        out.append({
            "character": key[0],
            "state": key[1],
            "image": key[2],
        })
    return out


def _packet_ids(packet: dict) -> list[str]:
    return [
        str(entry.get("id"))
        for entry in packet.get("selected_entries") or []
        if entry.get("id")
    ]
