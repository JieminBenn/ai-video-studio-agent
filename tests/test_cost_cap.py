"""Cost-cap enforcement (invariant #7).

The cap is honored at two layers: the project exposes a pre-spend budget guard, and
the orchestrator refuses to start or auto-advance into a paid stage once the cap is
spent. Because real costs are only known after a generation returns, at most the one
generation that crosses the cap completes; subsequent ones are blocked.
"""

import json

import pytest

from studio_agent.orchestrator.project import CostCapExceeded, Project
from studio_agent.orchestrator.state_machine import StateMachine
from studio_agent.prompt_approvals import confirm_prompt_batch
from studio_agent.providers.fake import FakeImageGen, FakeLLM, FakeMusic, FakeTTS, FakeVideoGen
from studio_agent.stages.base import Providers, Stage, StageResult
from studio_agent.stages.audio import AudioStage
from studio_agent.stages.bible import BibleStage
from studio_agent.stages.storyboard import StoryboardStage
from studio_agent.stages.video import VideoStage


def _project(tmp_path, *, cap, stages):
    return Project.create("cap test", root=tmp_path, stages=stages, cost_cap=cap)


# --- project-level budget guard -------------------------------------------------

def test_within_cost_cap_with_no_cap_is_always_true(tmp_path):
    p = Project.create("no cap", root=tmp_path, stages=["x"], cost_cap=None)
    p.add_cost(stage="x", provider="p", cost_usd=999.0, seconds=1.0)
    assert p.within_cost_cap() is True
    p.assert_budget_available()  # does not raise


def test_assert_budget_available_raises_once_cap_is_spent(tmp_path):
    p = _project(tmp_path, cap=0.10, stages=["x"])
    p.add_cost(stage="x", provider="p", cost_usd=0.08, seconds=1.0)
    p.assert_budget_available()  # 0.08 <= 0.10, still ok

    p.add_cost(stage="x", provider="p", cost_usd=0.05, seconds=1.0)  # now 0.13 > 0.10
    with pytest.raises(CostCapExceeded):
        p.assert_budget_available()


def test_assert_budget_available_raises_when_cap_is_exactly_spent(tmp_path):
    p = _project(tmp_path, cap=0.10, stages=["x"])
    p.add_cost(stage="x", provider="p", cost_usd=0.10, seconds=1.0)

    assert p.within_cost_cap() is False
    with pytest.raises(CostCapExceeded):
        p.assert_budget_available()


# --- orchestrator enforcement ---------------------------------------------------

class _CostStage(Stage):
    """A stage that records a fixed cost when it runs."""

    def __init__(self, name, cost):
        self.name = name
        self._cost = cost

    def run(self, project, providers):
        project.add_cost(stage=self.name, provider="fake", cost_usd=self._cost, seconds=1.0)
        return StageResult(status="complete")


def test_run_halts_before_a_stage_when_cap_already_spent(tmp_path):
    p = _project(tmp_path, cap=0.10, stages=["a", "b"])
    p.add_cost(stage="seed", provider="p", cost_usd=0.20, seconds=1.0)  # already over
    machine = StateMachine([_CostStage("a", 0.05), _CostStage("b", 0.05)])

    result = machine.run(p, Providers(), auto=True)

    assert result.cost_capped is True
    assert result.done is False
    assert p.stage_status("a") == "pending"  # never ran


def test_run_halts_after_a_stage_that_crosses_the_cap(tmp_path):
    p = _project(tmp_path, cap=0.10, stages=["a", "b"])
    machine = StateMachine([_CostStage("a", 0.15), _CostStage("b", 0.05)])

    result = machine.run(p, Providers(), auto=True)

    assert result.cost_capped is True
    assert result.done is False
    assert p.stage_status("a") == "complete"   # it ran and logged cost
    assert p.stage_status("b") == "pending"    # but we did not advance into b


# --- per-generation guard inside a paid loop ------------------------------------

class CostedVideoGen(FakeVideoGen):
    def __init__(self, cost):
        self._cost = cost

    def generate(self, prompt, *, out_path, **kwargs):
        gen = super().generate(prompt, out_path=out_path, **kwargs)
        gen.cost_usd = self._cost
        return gen


SHOTS = {
    "shots": [
        {"id": "sh-001", "scene": 1, "description": "wide", "camera": "wide", "action": "open",
         "dialogue": [], "duration_s": 2.0, "characters": ["Mara"], "deps": [],
         "reference_seed": 11, "keyframe": "sh-001.png"},
        {"id": "sh-002", "scene": 1, "description": "cu", "camera": "close-up", "action": "react",
         "dialogue": [], "duration_s": 2.0, "characters": ["Mara"], "deps": ["sh-001"],
         "reference_seed": 11, "keyframe": "sh-002.png"},
    ]
}


def _approve_video_prompts(project):
    for shot in SHOTS["shots"]:
        project.path(
            "storyboard", "prompts", f"{shot['id']}.video.md"
        ).write_text(
            f"{shot['id']} performs the visible action while the camera moves deliberately."
        )
    confirm_prompt_batch(project, "videos", confirmer="test")


def test_video_stage_blocks_the_generation_after_the_cap_is_crossed(tmp_path):
    p = Project.create("video cap", root=tmp_path, stages=["video"], cost_cap=0.05)
    p.path("storyboard", "shots.json").write_text(json.dumps(SHOTS))
    for shot in SHOTS["shots"]:
        FakeImageGen().generate(
            "kf", out_path=str(p.path("storyboard", "keyframes", shot["keyframe"])),
            seed=shot["reference_seed"],
        )
    _approve_video_prompts(p)

    # Each clip costs 0.10, cap is 0.05: clip one completes (cost only known after),
    # then the guard blocks clip two before it is generated.
    with pytest.raises(CostCapExceeded):
        VideoStage().run(p, Providers(video=CostedVideoGen(0.10)))

    assert p.path("assets", "clips", "sh-001.mp4").is_file()
    assert not p.path("assets", "clips", "sh-002.mp4").is_file()


class CostedLLM(FakeLLM):
    def __init__(self, cost):
        self._cost = cost

    def complete_json(self, prompt, *, system=None):
        gen = super().complete_json(prompt, system=system)
        gen.cost_usd = self._cost
        return gen


class CostedImageGen(FakeImageGen):
    def __init__(self, cost):
        self._cost = cost

    def generate(self, prompt, *, out_path, reference_images=None, **kwargs):
        gen = super().generate(
            prompt,
            out_path=out_path,
            reference_images=reference_images,
            **kwargs,
        )
        gen.cost_usd = self._cost
        return gen


class CostedTTS(FakeTTS):
    def __init__(self, cost):
        self._cost = cost

    def speak(self, text, *, out_path, **kwargs):
        gen = super().speak(text, out_path=out_path, **kwargs)
        gen.cost_usd = self._cost
        return gen


class CostedMusic(FakeMusic):
    def __init__(self, cost):
        self._cost = cost

    def compose(self, prompt, *, out_path, **kwargs):
        gen = super().compose(prompt, out_path=out_path, **kwargs)
        gen.cost_usd = self._cost
        return gen


def test_bible_stage_blocks_image_generation_after_cap_is_spent_by_llm(tmp_path):
    p = Project.create("bible cap", root=tmp_path, stages=["bible"], cost_cap=0.05)
    p.path("story", "plot.json").write_text(json.dumps({
        "themes": ["identity"],
        "characters": [{"name": "Mara", "role": "lead", "description": "pilot"}],
    }))

    with pytest.raises(CostCapExceeded):
        BibleStage().run(p, Providers(llm=CostedLLM(0.05), image=CostedImageGen(0.05)))

    cdir = p.path("bible", "characters", "mara")
    assert (cdir / "character.json").is_file()
    assert not (cdir / "reference.png").is_file()


def test_storyboard_stage_blocks_keyframes_after_cap_is_spent_by_llm(tmp_path):
    p = Project.create("storyboard cap", root=tmp_path, stages=["storyboard"], cost_cap=0.05)
    p.path("story", "script.json").write_text(json.dumps({
        "episodes": [{
            "episode": 1,
            "scenes": [{
                "scene": 1,
                "beats": ["open"],
                "dialogue": [{"character": "Mara", "line": "Go now."}],
            }],
        }]
    }))

    with pytest.raises(CostCapExceeded):
        StoryboardStage().run(
            p,
            Providers(llm=CostedLLM(0.05), image=CostedImageGen(0.05)),
        )

    assert p.path("storyboard", "shots.json").is_file()
    assert not p.path("storyboard", "keyframes", "sh-001.png").is_file()


def test_audio_stage_blocks_music_after_cap_is_spent_by_dialogue(tmp_path):
    p = Project.create(
        "audio cap", root=tmp_path, stages=["audio"], cost_cap=0.05,
        model_config={"music_enabled": True},
    )
    p.path("storyboard", "shots.json").write_text(json.dumps({
        "shots": [{
            "id": "sh-001",
            "duration_s": 2.0,
            "dialogue": [{"character": "Mara", "line": "Go now."}],
        }]
    }))
    p.path("story", "plot.json").write_text(json.dumps({"logline": "x", "themes": []}))

    with pytest.raises(CostCapExceeded):
        AudioStage().run(p, Providers(tts=CostedTTS(0.05), music=CostedMusic(0.05)))

    assert p.path("assets", "audio", "sh-001.dialogue.wav").is_file()
    assert not p.path("assets", "audio", "music.wav").is_file()


def test_native_audio_stage_extracts_without_tts_music_or_audio_cost(tmp_path):
    p = Project.create(
        "native audio cost",
        root=tmp_path,
        stages=["video", "audio"],
        cost_cap=0.05,
        model_config={"audio_mode": "native_video"},
    )
    p.path("storyboard", "shots.json").write_text(json.dumps(SHOTS))
    for shot in SHOTS["shots"]:
        FakeImageGen().generate(
            "kf",
            out_path=str(p.path("storyboard", "keyframes", shot["keyframe"])),
            seed=shot["reference_seed"],
        )
    _approve_video_prompts(p)
    providers = Providers(
        video=FakeVideoGen(),
        tts=CostedTTS(1.0),
        music=CostedMusic(1.0),
    )
    VideoStage().run(p, providers)
    costs_before = list(p.cost_log)

    result = AudioStage().run(p, providers)

    assert result.status == "complete"
    assert p.cost_log == costs_before
    assert not any(entry["stage"] == "audio" for entry in p.cost_log)
    assert p.path("assets", "audio", "sh-001.native.wav").is_file()
    assert p.path("assets", "audio", "sh-002.native.wav").is_file()
    assert not p.path("assets", "audio", "music.wav").exists()


def test_cli_report_explains_cost_cap_without_approve_instruction(tmp_path, capsys):
    from studio_agent import cli
    from studio_agent.orchestrator.state_machine import RunResult

    p = Project.create("report cap", root=tmp_path, stages=["video"], cost_cap=0.05)

    cli._report(p, RunResult(paused_at="video", cost_capped=True))

    out = capsys.readouterr().out
    assert "cost cap reached" in out
    assert "approve" not in out
