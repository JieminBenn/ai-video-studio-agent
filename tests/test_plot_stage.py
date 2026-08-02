"""Tests for the plot stage — idea.md -> plot.json via the LLM (idempotent)."""

import json

from studio_agent.orchestrator.project import Project
from studio_agent.providers.fake import FakeLLM
from studio_agent.reference_assets import save_reference_intake_uploads
from studio_agent.stages.base import Providers
from studio_agent.stages.plot import PlotStage

PIPELINE = ["plot", "script"]


class CountingFakeLLM(FakeLLM):
    def __init__(self):
        self.calls = 0

    def complete_json(self, prompt, *, system=None):
        self.calls += 1
        return super().complete_json(prompt, system=system)


def _project(tmp_path, idea="a lonely lighthouse keeper meets a talking gull", model_config=None):
    p = Project.create(idea, root=tmp_path, stages=PIPELINE, model_config=model_config)
    p.story_dir.joinpath("idea.md").write_text(idea)
    return p


def _scene_count(plot):
    return sum(len(ep["scenes"]) for ep in plot["arc"])


def test_plot_prompt_includes_locked_genre_when_set(tmp_path):
    p = _project(tmp_path, model_config={"genre": "东方奇幻"})

    class RecordingLLM(FakeLLM):
        def __init__(self):
            self.prompts = []

        def complete_json(self, prompt, *, system=None):
            self.prompts.append(prompt)
            return super().complete_json(prompt, system=system)

    llm = RecordingLLM()
    PlotStage().run(p, Providers(llm=llm))

    assert "GENRE" in llm.prompts[0]
    assert "东方奇幻" in llm.prompts[0]
    # 是什么: the persona itself carries a genre-specific expert identity.
    persona_line = llm.prompts[0].splitlines()[1]
    assert "东方奇幻" in persona_line and "specialist" in persona_line


def test_plot_stage_writes_plot_json_with_expected_shape(tmp_path):
    p = _project(tmp_path)
    providers = Providers(llm=FakeLLM())

    result = PlotStage().run(p, providers)

    assert result.status == "complete"
    plot = json.loads(p.path("story", "plot.json").read_text())
    for key in ["logline", "synopsis", "themes", "characters", "arc"]:
        assert key in plot
    # The fake is idea-aware: the idea text flows into the plot.
    assert "talking gull" in plot["logline"]


def test_plot_writes_brief_and_plot_from_one_llm_call(tmp_path):
    p = _project(tmp_path)
    llm = CountingFakeLLM()

    PlotStage().run(p, Providers(llm=llm))

    assert llm.calls == 1
    brief = json.loads(p.path("story", "creative_brief.json").read_text())
    assert brief["version"] == 2
    assert brief["source_idea"] == p.idea
    assert p.path("story", "plot.json").is_file()
    assert p.path("story", "visual_state_changes.json").is_file()


def test_short_film_plot_is_multi_scene(tmp_path):
    # M1 is a multi-scene short film; the fake plot must scale past a single scene
    # so the multi-scene orchestration is exercisable offline.
    p = _project(
        tmp_path,
        model_config={"product_format": {"name": "short_film", "target_duration_s": 180}},
    )
    PlotStage().run(p, Providers(llm=FakeLLM()))
    plot = json.loads(p.path("story", "plot.json").read_text())
    assert _scene_count(plot) >= 4
    assert "single self-contained scene" not in plot["synopsis"].lower()


def _scene_locations(plot):
    return [sc.get("location") for ep in plot["arc"] for sc in ep["scenes"]]


def test_short_film_plot_spans_multiple_locations(tmp_path):
    # A multi-scene short must visit more than one location, and reuse locations
    # across scenes so cross-scene location consistency is exercisable offline.
    p = _project(
        tmp_path,
        model_config={"product_format": {"name": "short_film", "target_duration_s": 180}},
    )
    PlotStage().run(p, Providers(llm=FakeLLM()))
    plot = json.loads(p.path("story", "plot.json").read_text())
    locations = _scene_locations(plot)
    assert all(locations), "every scene names a location"
    distinct = set(locations)
    assert len(distinct) >= 2  # more than one location
    assert len(distinct) < len(locations)  # at least one location reused across scenes


def test_unknown_format_falls_back_to_single_scene(tmp_path):
    # An unrecognized format gets the safe single-scene M0 vertical slice.
    p = _project(
        tmp_path,
        model_config={"product_format": {"name": "experimental_unlisted_format"}},
    )
    PlotStage().run(p, Providers(llm=FakeLLM()))
    plot = json.loads(p.path("story", "plot.json").read_text())
    assert _scene_count(plot) == 1


def test_plot_stage_logs_cost(tmp_path):
    p = _project(tmp_path)
    PlotStage().run(p, Providers(llm=FakeLLM()))
    assert any(entry["stage"] == "plot" for entry in p.cost_log)


def test_plot_stage_is_idempotent_no_duplicate_generation(tmp_path):
    p = _project(tmp_path)
    providers = Providers(llm=FakeLLM())

    PlotStage().run(p, providers)
    p.set_stage_status("plot", "complete")
    cost_entries_after_first = len(p.cost_log)

    # Re-running a completed stage must not regenerate or re-log cost.
    result = PlotStage().run(p, providers)
    assert result.status == "skipped"
    assert len(p.cost_log) == cost_entries_after_first


def test_plot_prompt_includes_reference_intent_context(tmp_path):
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
    PlotStage().run(p, Providers(llm=llm))

    assert "REFERENCE_INTENT:" in llm.prompts[0]
    assert "photo1 kisses photo2" in llm.prompts[0]
