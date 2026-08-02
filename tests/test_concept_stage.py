import json

from studio_agent.orchestrator.project import Project
from studio_agent.providers.fake import FakeLLM
from studio_agent.stages.base import Providers
from studio_agent.stages.concept import ConceptStage


def _project(tmp_path):
    project = Project.create(
        "a person walking in slow motion down a neon alley",
        root=tmp_path,
        stages=["concept", "bible", "clip", "video", "review", "assemble"],
        cost_cap=5.0,
        model_config={"language": "en", "product_format": {"name": "short_video", "mode": "clip"}},
    )
    project.story_dir.joinpath("idea.md").write_text(project.idea + "\n")
    project.save()
    return project


def test_concept_prompt_includes_locked_genre_when_set(tmp_path):
    project = _project(tmp_path)
    project.model_config["genre"] = "赛博朋克"

    class RecordingLLM(FakeLLM):
        def __init__(self):
            self.prompts = []

        def complete_json(self, prompt, *, system=None):
            self.prompts.append(prompt)
            return super().complete_json(prompt, system=system)

    llm = RecordingLLM()
    ConceptStage().run(project, Providers(llm=llm))

    assert "GENRE" in llm.prompts[0]
    assert "赛博朋克" in llm.prompts[0]
    persona_line = llm.prompts[0].splitlines()[1]
    assert "赛博朋克" in persona_line and "specialist" in persona_line


def test_concept_writes_bible_compatible_plot(tmp_path):
    project = _project(tmp_path)
    result = ConceptStage().run(project, Providers(llm=FakeLLM()))
    assert result.status == "complete"

    plot = json.loads(project.path("story", "plot.json").read_text())
    assert plot["characters"], "concept must produce at least one character"
    assert all("name" in c for c in plot["characters"])
    assert all("role" in c for c in plot["characters"])
    assert plot["locations"], "concept must produce a location for the bible"
    assert plot["locations"][0].get("scene") == 1
    assert plot["locations"][0]["name"] == "Setting", "fake location name must flow through"
    assert plot["themes"] == []  # non-narrative: no themes


def test_concept_writes_creative_brief_from_the_same_generation(tmp_path):
    project = _project(tmp_path)

    ConceptStage().run(project, Providers(llm=FakeLLM()))

    brief = json.loads(project.path("story", "creative_brief.json").read_text())
    assert brief["version"] == 2
    assert brief["source_idea"] == project.idea
    assert brief["visual_direction"]["camera_language"]["value"]


def test_concept_is_idempotent(tmp_path):
    project = _project(tmp_path)
    providers = Providers(llm=FakeLLM())
    ConceptStage().run(project, providers)
    project.set_stage_status("concept", "complete")
    again = ConceptStage().run(project, providers)
    assert again.status == "skipped"


def test_concept_chinese_language(tmp_path):
    project = Project.create(
        "一个人在霓虹灯小巷中慢动作行走",
        root=tmp_path,
        stages=["concept", "bible", "clip", "video", "review", "assemble"],
        cost_cap=5.0,
        model_config={"language": "zh", "product_format": {"name": "short_video", "mode": "clip"}},
    )
    project.story_dir.joinpath("idea.md").write_text(project.idea + "\n")
    project.save()

    result = ConceptStage().run(project, Providers(llm=FakeLLM()))
    assert result.status == "complete"

    plot = json.loads(project.path("story", "plot.json").read_text())
    assert plot["characters"], "concept must produce at least one character (zh)"
    assert plot["locations"][0].get("scene") == 1


def test_transformation_concept_separates_baseline_identity_from_visual_states(tmp_path):
    idea = "一个男人在马路上，身上冒出红色气体，变成一个狼人"
    project = Project.create(
        idea,
        root=tmp_path,
        stages=["concept", "bible", "clip", "video", "audio", "assemble"],
        cost_cap=5.0,
        model_config={"language": "zh", "product_format": {"name": "short_video", "mode": "clip"}},
    )
    project.story_dir.joinpath("idea.md").write_text(idea + "\n")

    ConceptStage().run(project, Providers(llm=FakeLLM()))

    plot = json.loads(project.path("story", "plot.json").read_text())
    character = plot["characters"][0]
    assert character["name"] == "男人"
    assert "正在变" not in character["name"] + character["description"]
    assert "红色气体" not in plot["locations"][0]["description"]

    states = json.loads(project.path("story", "visual_state_changes.json").read_text())
    plan = states["characters"][0]
    assert plan["initial_state"] == "human"
    assert [state["id"] for state in plan["states"]] == [
        "human",
        "gas-onset",
        "partial-werewolf",
        "werewolf",
    ]
