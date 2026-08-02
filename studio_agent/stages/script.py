"""Script stage: expand ``story/plot.json`` into ``story/script.json`` via the LLM.

The screenplay (episodes -> scenes -> beats -> dialogue) is grounded in the full plot
so long-form structure stays coherent (PLAN.md: RAG over the plot for global context).
Idempotent (invariant #3): skips without re-generating if already complete.
"""

from __future__ import annotations

import json

from .base import Providers, Stage, StageResult
from ..creative_brief import compact_creative_brief, load_creative_brief
from ..formats import format_prompt_context, genre_persona, genre_prompt, project_format
from ..language import language_label
from ..reference_assets import reference_intent_prompt

_PROMPT_TEMPLATE = """[task:script]
You are {genre_persona}a screenwriter writing for AI-video generation. Using the full plot below, write
a filmable screenplay as JSON. Preserve the requested project language and target format.

Return only valid JSON with keys `episodes` (list) and `structure_review` (object).
Each episode has `scenes`; each scene has:
- scene: scene number.
- heading: INT./EXT. location - time.
- objective: what the scene changes in the story.
- beats: 3-6 concrete visual action beats, each physically observable on screen.
- dialogue: list of {{character, line, subtext}}. Dialogue should be short, playable,
  and grounded in character desire.
- narration: off-screen narrator voiceover (旁白) for this scene, or "" when the scene is
  not narrated. This is the narrator's spoken line, NOT character dialogue and NOT a stage
  direction; write it only when the story is told with a narrating voice.
- imageable_moments: 2-4 moments that should become keyframes or memorable shots.
- emotional_turn: the before/after emotional shift of the scene.

`structure_review` audits the script the way a human reviewer does and has:
- coverage: a list with exactly four entries, one per dramatic stage 起 (setup/引入),
  承 (development/推进), 转 (turn/转折), 合 (resolution/升华). Each has stage, label,
  covered (bool), and scenes (the scene numbers that fulfill that stage).
- executability: one entry per scene with scene, executable (bool — can a director render
  an image directly from the scene's description), and a short note.
- complete: true only when all four 起承转合 stages are covered.
- summary: one line on structural completeness (结构完整性) and any gaps.

Quality bar:
- Avoid placeholder dialogue such as "We begin" or "And so it does."
- Every beat must include physical action, blocking, props, or environment change.
- Narration is a narrator's voiceover laid over the picture, never lip-synced to a
  character; keep it concise and leave it "" unless the piece is genuinely narrated.
- Write subtext; do not explain the theme out loud.
- Keep scenes coherent for storyboard planning: setup, escalation, choice, consequence.

IDEA: {idea}
LANGUAGE: {language}
FORMAT: {format}
{genre}{reference_intent}
CREATIVE_BRIEF: {creative_brief}
PLOT: {plot}
"""


def _render_structure_review_md(review: dict) -> str:
    """Render the script's 起承转合 + 可执行性 review as a human-readable gate artifact."""
    lines = ["# 剧本结构审查 / Script structure review", ""]
    lines.append("## 起承转合 / Structure completeness")
    for entry in review.get("coverage", []):
        stage = entry.get("stage", "")
        label = entry.get("label", "")
        scenes = entry.get("scenes") or []
        if entry.get("covered"):
            where = ", ".join(str(s) for s in scenes)
            lines.append(f"- {stage} ({label}): ✅ scene(s) {where}")
        else:
            lines.append(f"- {stage} ({label}): ⚠️ not yet covered")

    lines.append("")
    lines.append("## 可执行性 / Executability (renderable as images)")
    executability = review.get("executability") or []
    if executability:
        for item in executability:
            scene = item.get("scene")
            mark = "✅" if item.get("executable") else "⚠️"
            note = str(item.get("note") or "").strip()
            suffix = f" — {note}" if note else ""
            lines.append(f"- Scene {scene}: {mark}{suffix}")
    else:
        lines.append("- (no scenes)")

    summary = str(review.get("summary") or "").strip()
    if summary:
        lines.extend(["", f"**Summary / 总评:** {summary}"])
    return "\n".join(lines) + "\n"


class ScriptStage(Stage):
    name = "script"

    def run(self, project, providers: Providers) -> StageResult:
        if project.stage_status(self.name) == "complete":
            return StageResult(status="skipped", message="script already complete")

        idea = project.story_dir.joinpath("idea.md").read_text().strip()
        plot = project.path("story", "plot.json").read_text().strip()
        # PLOT must be one line so the fake/real LLM can parse it back deterministically.
        prompt = _PROMPT_TEMPLATE.format(
            idea=idea,
            genre_persona=genre_persona(project.model_config),
            language=language_label(project.model_config.get("language", "en")),
            format=format_prompt_context(project_format(project)),
            genre=genre_prompt(project.model_config),
            reference_intent=reference_intent_prompt(project),
            creative_brief=json.dumps(
                compact_creative_brief(load_creative_brief(project)),
                ensure_ascii=False,
            ),
            plot=json.dumps(json.loads(plot), ensure_ascii=False),
        )

        project.assert_budget_available()
        gen = providers.llm.complete_json(prompt)
        out = project.path("story", "script.json")
        out.write_text(json.dumps(gen.content, indent=2))

        # Surface the 起承转合 + 可执行性 review at the human script gate (invariant #2).
        review = gen.content.get("structure_review") if isinstance(gen.content, dict) else None
        if isinstance(review, dict):
            project.path("story", "script.review.md").write_text(
                _render_structure_review_md(review)
            )

        project.add_generation_cost(stage=self.name, generation=gen)
        return StageResult(status="complete", message=f"wrote {out.name}")
