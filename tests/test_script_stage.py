"""Tests for the script stage — plot.json -> script.json via the LLM (idempotent)."""

import json

from studio_agent.orchestrator.project import Project
from studio_agent.providers.fake import FakeLLM
from studio_agent.reference_assets import save_reference_intake_uploads
from studio_agent.stages.base import Providers
from studio_agent.stages.script import ScriptStage

PIPELINE = ["plot", "script"]

PLOT = {
    "logline": "A short film about a clockmaker who repairs memories.",
    "synopsis": "A clockmaker discovers a watch that mends the past.",
    "themes": ["memory", "regret"],
    "characters": [
        {"name": "Mara", "role": "lead", "description": "the clockmaker"},
        {"name": "The Customer", "role": "supporting", "description": "a grieving man"},
    ],
    "arc": [
        {"episode": 1, "scenes": [
            {"scene": 1, "summary": "Mara opens the shop at dawn."},
            {"scene": 2, "summary": "The customer brings a broken watch."},
        ]},
    ],
}


def _project(tmp_path):
    p = Project.create("a clockmaker who repairs memories", root=tmp_path, stages=PIPELINE)
    p.story_dir.joinpath("idea.md").write_text("a clockmaker who repairs memories")
    p.path("story", "plot.json").write_text(json.dumps(PLOT))
    return p


def test_script_stage_writes_script_json_with_expected_shape(tmp_path):
    p = _project(tmp_path)

    result = ScriptStage().run(p, Providers(llm=FakeLLM()))

    assert result.status == "complete"
    script = json.loads(p.path("story", "script.json").read_text())
    assert "episodes" in script
    scene = script["episodes"][0]["scenes"][0]
    for key in ["scene", "heading", "beats", "dialogue"]:
        assert key in scene


def test_script_scenes_carry_a_narration_voiceover_field(tmp_path):
    p = _project(tmp_path)
    ScriptStage().run(p, Providers(llm=FakeLLM()))
    script = json.loads(p.path("story", "script.json").read_text())

    scenes = [sc for ep in script["episodes"] for sc in ep["scenes"]]
    assert scenes, "expected at least one scene"
    for scene in scenes:
        assert "narration" in scene
        assert isinstance(scene["narration"], str)
    # At least one scene actually carries off-screen narration text.
    assert any(sc["narration"].strip() for sc in scenes)


def test_script_is_grounded_in_the_plot(tmp_path):
    p = _project(tmp_path)
    ScriptStage().run(p, Providers(llm=FakeLLM()))
    script = json.loads(p.path("story", "script.json").read_text())

    # One script scene per plot scene, and dialogue uses a plot character.
    plot_scene_count = sum(len(ep["scenes"]) for ep in PLOT["arc"])
    script_scene_count = sum(len(ep["scenes"]) for ep in script["episodes"])
    assert script_scene_count == plot_scene_count

    speakers = {
        line["character"]
        for ep in script["episodes"] for sc in ep["scenes"] for line in sc["dialogue"]
    }
    plot_names = {c["name"] for c in PLOT["characters"]}
    assert speakers & plot_names  # at least one real character speaks


def test_script_includes_a_qcz_structure_review(tmp_path):
    p = _project(tmp_path)
    ScriptStage().run(p, Providers(llm=FakeLLM()))
    script = json.loads(p.path("story", "script.json").read_text())

    review = script["structure_review"]
    stages = [c["stage"] for c in review["coverage"]]
    for stage in ["起", "承", "转", "合"]:
        assert stage in stages
    # Per-scene executability (可执行性) for each script scene.
    scene_count = sum(len(ep["scenes"]) for ep in script["episodes"])
    assert len(review["executability"]) == scene_count


def test_script_stage_writes_human_readable_structure_review(tmp_path):
    p = _project(tmp_path)
    ScriptStage().run(p, Providers(llm=FakeLLM()))

    md = p.path("story", "script.review.md")
    assert md.is_file()
    text = md.read_text()
    assert text.strip()
    for stage in ["起", "承", "转", "合"]:
        assert stage in text
    assert "可执行性" in text or "executab" in text.lower()


def test_script_stage_logs_cost(tmp_path):
    p = _project(tmp_path)
    ScriptStage().run(p, Providers(llm=FakeLLM()))
    assert any(entry["stage"] == "script" for entry in p.cost_log)


def test_script_stage_is_idempotent(tmp_path):
    p = _project(tmp_path)
    providers = Providers(llm=FakeLLM())

    ScriptStage().run(p, providers)
    p.set_stage_status("script", "complete")
    cost_after_first = len(p.cost_log)

    result = ScriptStage().run(p, providers)
    assert result.status == "skipped"
    assert len(p.cost_log) == cost_after_first


def test_script_prompt_includes_reference_intent_context(tmp_path):
    p = _project(tmp_path)
    save_reference_intake_uploads(
        p,
        [
            {"filename": "hero.png", "data": b"\x89PNG\r\n\x1a\nhero", "content_type": "image/png"},
            {"filename": "villain.png", "data": b"\x89PNG\r\n\x1a\nvillain", "content_type": "image/png"},
        ],
        note="photo1 kisses photo2",
    )

    class RecordingLLM(FakeLLM):
        def __init__(self):
            self.prompts = []

        def complete_json(self, prompt, *, system=None):
            self.prompts.append(prompt)
            return super().complete_json(prompt, system=system)

    llm = RecordingLLM()
    ScriptStage().run(p, Providers(llm=llm))

    assert "REFERENCE_INTENT:" in llm.prompts[0]
    assert "photo1 kisses photo2" in llm.prompts[0]
    assert llm.prompts[0].rstrip().endswith("}")


def test_script_prompt_includes_locked_genre_when_set(tmp_path):
    p = Project.create("a clockmaker", root=tmp_path, stages=PIPELINE,
                       model_config={"genre": "中国古代神话"})
    p.story_dir.joinpath("idea.md").write_text("a clockmaker who repairs memories")
    p.path("story", "plot.json").write_text(json.dumps(PLOT))

    class RecordingLLM(FakeLLM):
        def __init__(self):
            self.prompts = []

        def complete_json(self, prompt, *, system=None):
            self.prompts.append(prompt)
            return super().complete_json(prompt, system=system)

    llm = RecordingLLM()
    ScriptStage().run(p, Providers(llm=llm))

    assert "GENRE" in llm.prompts[0]
    assert "中国古代神话" in llm.prompts[0]
    persona_line = llm.prompts[0].splitlines()[1]
    assert "中国古代神话" in persona_line and "specialist" in persona_line


def test_script_prompt_omits_genre_line_when_unset(tmp_path):
    p = _project(tmp_path)

    class RecordingLLM(FakeLLM):
        def __init__(self):
            self.prompts = []

        def complete_json(self, prompt, *, system=None):
            self.prompts.append(prompt)
            return super().complete_json(prompt, system=system)

    llm = RecordingLLM()
    ScriptStage().run(p, Providers(llm=llm))

    assert "GENRE (" not in llm.prompts[0]


def test_script_prompt_includes_creative_brief(tmp_path):
    p = _project(tmp_path)
    p.path("story", "creative_brief.json").write_text(json.dumps({
        "version": 2,
        "visual_direction": {
            "tone": {"value": "tender dread"},
            "camera_language": {"value": "patient and intimate"},
        },
        "hard_avoidances": ["flat lighting"],
    }))

    class RecordingLLM(FakeLLM):
        def __init__(self):
            self.prompts = []

        def complete_json(self, prompt, *, system=None):
            self.prompts.append(prompt)
            return super().complete_json(prompt, system=system)

    llm = RecordingLLM()
    ScriptStage().run(p, Providers(llm=llm))

    assert 'CREATIVE_BRIEF: {"tone": "tender dread"' in llm.prompts[0]
    assert "flat lighting" in llm.prompts[0]
