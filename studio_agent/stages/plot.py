"""Plot stage: expand ``story/idea.md`` into ``story/plot.json`` via the LLM.

Idempotent (invariant #3): if the stage is already complete, it skips without
re-generating or re-logging cost. The exact prompt is built from the idea and tagged
so the (fake or real) LLM can produce a structured plot.
"""

from __future__ import annotations

import json

from .base import Providers, Stage, StageResult
from ..creative_brief import normalize_creative_brief
from ..formats import format_prompt_context, genre_persona, genre_prompt, project_format
from ..language import language_label
from ..reference_assets import reference_intent_prompt
from ..visual_states import normalize_visual_state_changes

_PROMPT_TEMPLATE = """[task:plot]
You are {genre_persona}a showrunner developing a filmable AI-video short. Expand the idea into a
specific structured plot, not generic filler. Preserve the requested project language.

Return only valid JSON with keys `creative_brief`, `plot`, and
`visual_state_changes`.
`creative_brief` contains intent_summary, hard_avoidances, and visual_direction with
tone, visual_world, camera_language, light_texture, and character_treatment.
`plot` contains:
- logline: one vivid sentence with protagonist, central conflict, and hook.
- synopsis: 2-4 paragraphs with concrete cause/effect story progression.
- themes: 2-5 precise themes.
- characters: list of named characters, each with name, role, description, visual_hook,
  desire, fear_or_flaw, and relationship_to_conflict. Do not use placeholder names like
  "Protagonist" unless the idea explicitly asks for that name. Describe each character's
  clean baseline identity; do not bake transformations, temporary effects, injuries,
  disguises, aging, or later forms into the canonical name/description.
- arc: list of episodes, each with scenes. Each scene needs scene, heading, summary,
  emotional_turn, visual_set_pieces (visual set pieces), and story_function.

Quality bar:
- Make every scene imageable by a director and useful for later image/video prompts.
- Include at least one visual set piece that could become a memorable shot.
- Give the lead a clear desire, obstacle, choice, and emotional turn.
- Avoid vague phrases such as "a small complication raises the stakes" unless you name
  exactly what happens on screen.
`visual_state_changes` uses the same character names as `plot.characters`. Return a
`characters` list only for time-varying appearance. Each plan has character,
initial_state, states, and transitions. States have id, label, kind (`base`, `transient`,
or `endpoint`), description, appearance_changes (a list of material identity/appearance
deltas), and reference_required. Base states require a reference. Endpoint states require
one only when appearance_changes is nonempty. Location, pose, action, lighting,
inside/outside a screen, 2D/3D context, camera context, and render context alone are not
material appearance changes: use appearance_changes: [] and reference_required: false.
Transitions have from, to, trigger,
ordered_state_ids, and preserve. Return {{"characters": []}} when no change occurs.

IDEA: {idea}
LANGUAGE: {language}
FORMAT: {format}
{genre}{reference_intent}
"""


class PlotStage(Stage):
    name = "plot"

    def run(self, project, providers: Providers) -> StageResult:
        if project.stage_status(self.name) == "complete":
            return StageResult(status="skipped", message="plot already complete")

        idea = project.story_dir.joinpath("idea.md").read_text().strip()
        language = project.model_config.get("language", "en")
        prompt = _PROMPT_TEMPLATE.format(
            idea=idea,
            genre_persona=genre_persona(project.model_config),
            language=language_label(language),
            format=format_prompt_context(project_format(project)),
            genre=genre_prompt(project.model_config),
            reference_intent=reference_intent_prompt(project),
        )

        project.assert_budget_available()
        gen = providers.llm.complete_json(prompt)
        content = gen.content if isinstance(gen.content, dict) else {}
        plot = content.get("plot") if isinstance(content.get("plot"), dict) else content

        brief_path = project.path("story", "creative_brief.json")
        if not brief_path.is_file():
            brief = normalize_creative_brief(
                content.get("creative_brief"),
                idea=idea,
                model_config=project.model_config,
                user_inputs=project.model_config.get("creative_inputs"),
            )
            brief_path.write_text(json.dumps(brief, ensure_ascii=False, indent=2) + "\n")
        out = project.path("story", "plot.json")
        out.write_text(json.dumps(plot, ensure_ascii=False, indent=2) + "\n")
        states_path = project.path("story", "visual_state_changes.json")
        if not states_path.is_file():
            states = normalize_visual_state_changes(
                content.get("visual_state_changes"),
                plot.get("characters") or [],
            )
            states_path.write_text(
                json.dumps(states, ensure_ascii=False, indent=2) + "\n"
            )

        project.add_generation_cost(stage=self.name, generation=gen)
        return StageResult(status="complete", message=f"wrote {out.name}")
