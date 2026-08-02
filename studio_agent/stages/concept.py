"""Concept stage (clip mode): idea -> characters + location, no narrative.

Replaces plot+script for short-video (clip) mode. Turns the idea into a small cast and
one location and writes ``story/plot.json`` in the exact shape the bible already reads,
so the unchanged BibleStage builds the full reference-image set for each character and
location (invariant #4). No plot arc, no dialogue.

Writes::

    story/plot.json   characters + locations (scene 1), empty narrative fields

Idempotent (invariant #3): skips when already complete.
"""

from __future__ import annotations

import json

from .base import Providers, Stage, StageResult
from ..creative_brief import normalize_creative_brief
from ..formats import genre_persona, genre_prompt
from ..language import language_label
from ..visual_states import normalize_visual_state_changes

_PROMPT_TEMPLATE = """[task:concept]
You are {genre_persona}a commercial director planning a single striking short-video shot (think a
high-end social-media / 抖音 clip), NOT a story. Turn the idea into the on-screen subject(s)
and the setting. No plot, no dialogue, no narrative arc.

Return only valid JSON with keys `creative_brief`, `concept`, and
`visual_state_changes`.
`creative_brief` contains intent_summary, hard_avoidances, and visual_direction with
tone, visual_world, camera_language, light_texture, and character_treatment.
`concept` contains:
- characters: list of the on-screen subject(s), each with name, role, description. Use a
  concrete BASELINE description of appearance/wardrobe, not a backstory. Usually one
  subject. Do not put transformations, temporary effects, damage, or later forms in the
  character name or baseline description.
- location: object with name and description of the clean reusable setting. Do not put
  the actor, temporary effects, or one-time action into the location description.
`visual_state_changes` contains a `characters` list. For each character whose appearance
changes over time, return character, initial_state, ordered states, and transitions.
Each state has id, label, kind (`base`, `transient`, or `endpoint`), description,
appearance_changes (a list of material identity/appearance deltas), and
reference_required. Base states require references. Endpoint states require one only
when appearance_changes is nonempty. Location, pose, action, lighting, inside/outside a
screen, 2D/3D context, camera context, and render context alone use
appearance_changes: [] and reference_required: false. Transient states do not require
references. Each transition has from, to, trigger, ordered_state_ids, and preserve.
Return {{"characters": []}} when no appearance change occurs.

IDEA: {idea}
LANGUAGE: {language}
{genre}"""


class ConceptStage(Stage):
    name = "concept"

    def run(self, project, providers: Providers) -> StageResult:
        if project.stage_status(self.name) == "complete":
            return StageResult(status="skipped", message="concept already complete")

        idea = project.story_dir.joinpath("idea.md").read_text().strip()
        language = project.model_config.get("language", "en")
        prompt = _PROMPT_TEMPLATE.format(
            idea=idea,
            genre_persona=genre_persona(project.model_config),
            language=language_label(language),
            genre=genre_prompt(project.model_config),
        )

        project.assert_budget_available()
        gen = providers.llm.complete_json(prompt)
        bundle = gen.content if isinstance(gen.content, dict) else {}
        content = bundle.get("concept") if isinstance(bundle.get("concept"), dict) else bundle

        brief_path = project.path("story", "creative_brief.json")
        if not brief_path.is_file():
            brief = normalize_creative_brief(
                bundle.get("creative_brief"),
                idea=idea,
                model_config=project.model_config,
                user_inputs=project.model_config.get("creative_inputs"),
            )
            brief_path.write_text(json.dumps(brief, ensure_ascii=False, indent=2) + "\n")

        characters = content.get("characters") or []
        location = content.get("location") or {}
        location_name = location.get("name") or "Setting"
        plot = {
            "logline": content.get("logline", f"A short-video shot of {idea}."),
            "synopsis": "",
            "themes": [],
            "characters": characters,
            "locations": [{
                "name": location_name,
                "description": location.get("description", ""),
                "scene": 1,
            }],
            "arc": [],
        }
        states_path = project.path("story", "visual_state_changes.json")
        if not states_path.is_file():
            states = normalize_visual_state_changes(
                bundle.get("visual_state_changes"),
                characters,
            )
            states_path.write_text(
                json.dumps(states, ensure_ascii=False, indent=2) + "\n"
            )
        out = project.path("story", "plot.json")
        out.write_text(json.dumps(plot, ensure_ascii=False, indent=2) + "\n")

        project.add_generation_cost(stage=self.name, generation=gen)
        return StageResult(
            status="complete",
            message=f"concept: {len(characters)} subject(s), 1 location",
        )
