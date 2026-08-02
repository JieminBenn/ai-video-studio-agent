import json
from pathlib import Path

import pytest

from studio_agent.orchestrator.project import Project
from studio_agent.prompt_approvals import confirm_prompt_batch
from studio_agent.providers.fake import FakeImageGen, FakeLLM, FakeReferenceAnalyzer, FakeVideoGen
from studio_agent.stages.base import Providers
from studio_agent.stages.bible import BibleStage
from studio_agent.stages.concept import ConceptStage
from studio_agent.stages.clip import ClipStage, _PROMPT_TEMPLATE
from studio_agent.stages.keyframes import KeyframesStage


def _render_approved_keyframes(project, providers):
    confirm_prompt_batch(project, "keyframes", confirmer="test")
    return KeyframesStage().run(project, providers)


def test_clip_prompt_requests_complexity_and_allows_few_beats():
    text = _PROMPT_TEMPLATE
    # The model must name a complexity tier matching the clamp in clip_sequence.
    assert "complexity" in text
    assert "simple" in text and "moderate" in text and "complex" in text
    # A simple idea must be allowed to be ONE short clip with very few beats.
    assert "1-2" in text or "1–2" in text  # ascii or en-dash hyphen
    # The rigid 3-8 floor is gone.
    assert "3-8 ordered objects" not in text


def test_clip_planner_template_is_restraint_first_and_threads_camera_movement_skill():
    # Clip mode must get the same restrained, motivated camera-movement selection as story
    # mode: one move, default to the smallest/static, reserve orbit/crane for earned beats —
    # and the camera_movement rulebook injected the same way story mode threads it.
    low = _PROMPT_TEMPLATE.lower()
    assert "one" in low and "move" in low
    assert "reserve" in low   # reserve orbit/crane/aerial/whip for earned beats
    assert "static" in low    # static/locked-off is a valid choice
    assert "{camera_movement_skill}" in _PROMPT_TEMPLATE


def test_clip_planner_prompt_carries_camera_movement_skill(tmp_path):
    project, providers = _clip_project(tmp_path)

    class RecordingLLM(FakeLLM):
        def __init__(self):
            self.prompts = []

        def complete_json(self, prompt, *, system=None):
            self.prompts.append(prompt)
            return super().complete_json(prompt, system=system)

    llm = RecordingLLM()
    ClipStage().run(project, Providers(llm=llm, image=FakeImageGen()))

    planning = [pr for pr in llm.prompts if "[task:clip]" in pr]
    assert planning
    # Real skill text injected (heading of skills/camera_movement.md), not "(none)".
    assert "Camera Movement Skill" in planning[0]


class DialogueLLM(FakeLLM):
    def complete_json(self, prompt, *, system=None):
        gen = super().complete_json(prompt, system=system)
        if "[task:clip]" in prompt:
            if "LANGUAGE: Chinese" in prompt:
                gen.content["dialogue"] = [{"character": "主角", "line": "别回头。"}]
            else:
                gen.content["dialogue"] = [{"character": "Subject", "line": "Watch this."}]
        return gen


class CountingLLM(FakeLLM):
    def __init__(self):
        self.clip_calls = 0

    def complete_json(self, prompt, *, system=None):
        if "[task:clip]" in prompt:
            self.clip_calls += 1
        return super().complete_json(prompt, system=system)


def _clip_project(
    tmp_path,
    clip_count=1,
    clip_seconds=15,
    language="en",
    idea="a person walking in slow motion down a neon alley",
):
    project = Project.create(
        idea,
        root=tmp_path,
        stages=["concept", "bible", "clip", "video", "review", "assemble"],
        cost_cap=50.0,
        model_config={
            "language": language,
            "product_format": {"name": "short_video", "mode": "clip"},
            "clip_count": clip_count,
            "clip_seconds": clip_seconds,
        },
    )
    project.story_dir.joinpath("idea.md").write_text(project.idea + "\n")
    project.save()
    providers = Providers(llm=FakeLLM(), image=FakeImageGen())
    ConceptStage().run(project, providers)
    BibleStage().run(project, providers)
    return project, providers


def test_clip_writes_single_video_compatible_shot(tmp_path):
    project, providers = _clip_project(tmp_path, clip_count=1, clip_seconds=15)
    result = ClipStage().run(project, providers)
    _render_approved_keyframes(project, providers)
    assert result.status == "complete"

    shots = json.loads(project.path("storyboard", "shots.json").read_text())["shots"]
    assert len(shots) == 1
    shot = shots[0]
    assert shot["dialogue"] == []                      # non-narrative
    assert shot["duration_s"] == 15                    # honors clip_seconds
    assert shot["reference_characters"], "must condition on bible characters"
    assert shot["movement_motivation"]
    assert shot["start_frame"] and shot["end_frame"]
    assert shot["craft_recipe_ids"]
    assert project.path("knowledge", "packets", "shot-sh-001.json").is_file()
    # keyframe was rendered through the inherited storyboard logic
    assert project.path("storyboard", "keyframes", shot["keyframe"]).is_file()


def test_clip_beats_carry_secondary_motion_and_per_beat_duration(tmp_path):
    # Run a fake clip-mode pipeline and read the compiled shots; the beats must carry
    # secondary motion and a per-beat duration_s (which the compiler turns into proportional
    # timing — that rendering is covered directly in test_video_prompt.py).
    project, providers = _clip_project(tmp_path, clip_count=1, clip_seconds=15)
    result = ClipStage().run(project, providers)
    assert result.status == "complete"

    shots = json.loads(project.path("storyboard", "shots.json").read_text())["shots"]
    beats = shots[0]["visual_beats"]
    assert beats, "clip planning must produce beats"
    for beat in beats:
        assert str(beat.get("secondary_motion") or "").strip(), (
            f"beat {beat.get('id')} is missing secondary_motion"
        )
        assert beat.get("duration_s")


def test_clip_count_two_chains_clips(tmp_path):
    project, providers = _clip_project(tmp_path, clip_count=2, clip_seconds=15)
    ClipStage().run(project, providers)
    shots = json.loads(project.path("storyboard", "shots.json").read_text())["shots"]
    assert len(shots) == 2
    assert shots[1]["deps"] == [shots[0]["id"]]        # clip 2 continues clip 1
    assert sum(s["duration_s"] for s in shots) == 30


def test_clip_plans_one_global_transformation_sequence_then_partitions_it(tmp_path):
    project, providers = _clip_project(
        tmp_path,
        language="zh",
        idea="一个男人在马路上，身上冒出红色气体，变成一个狼人",
    )
    project.model_config["clip_target_duration_s"] = 30
    counter = CountingLLM()
    providers.llm = counter

    ClipStage().run(project, providers)

    sequence = json.loads(project.path("storyboard", "visual_sequence.json").read_text())
    shots = json.loads(project.path("storyboard", "shots.json").read_text())["shots"]
    assert counter.clip_calls == 1
    assert sequence["intent_class"] == "transformation"
    assert sequence["target_duration_s"] == 30
    assert [shot["duration_s"] for shot in shots] == [15, 15]
    assert [beat["id"] for shot in shots for beat in shot["visual_beats"]] == [
        "human",
        "gas-onset",
        "partial-werewolf",
        "werewolf-reveal",
    ]
    assert shots[0]["start_states"] == {"男人": "human"}
    assert shots[0]["end_states"] == {"男人": "gas-onset"}
    assert shots[1]["start_states"] == {"男人": "gas-onset"}
    assert shots[1]["end_states"] == {"男人": "werewolf"}
    assert any(path.endswith("states/werewolf/reference.png") for path in shots[1]["state_reference_images"])
    assert any(
        ref["character"] == "男人"
        and ref["state"] == "werewolf"
        and ref["image"].endswith("states/werewolf/reference.png")
        for ref in shots[1]["target_state_references"]
    )
    assert any(
        path.endswith("states/werewolf/reference.png")
        for path in shots[1]["target_state_reference_images"]
    )
    assert not any(path.endswith("states/werewolf/reference.png") for path in shots[1]["keyframe_reference_images"])


class SeedanceRangeVideo(FakeVideoGen):
    """Fake video provider that declares Seedance's [4, 15] duration band."""

    @property
    def capabilities(self):
        from dataclasses import replace

        return replace(super().capabilities, min_duration_s=4, max_duration_s=15)


def test_clip_clamps_duration_to_provider_range(tmp_path):
    # The floor/ceiling come from the video provider's declared capabilities.
    project, providers = _clip_project(tmp_path, clip_count=1, clip_seconds=2)
    providers = Providers(llm=providers.llm, image=providers.image, video=SeedanceRangeVideo())
    ClipStage().run(project, providers)
    shots = json.loads(project.path("storyboard", "shots.json").read_text())["shots"]
    assert shots[0]["duration_s"] == 4                 # clamped up to the provider's 4s floor


def test_clip_keeps_dialogue_only_when_idea_requests_speech(tmp_path):
    project, providers = _clip_project(tmp_path)
    project.idea = 'Subject looks into camera and says "Watch this."'
    project.story_dir.joinpath("idea.md").write_text(project.idea)
    providers.llm = DialogueLLM()

    ClipStage().run(project, providers)

    shot = json.loads(project.path("storyboard", "shots.json").read_text())["shots"][0]
    assert shot["dialogue"] == [{"character": "Subject", "line": "Watch this."}]


def test_clip_preserves_requested_chinese_dialogue(tmp_path):
    project, providers = _clip_project(tmp_path, language="zh")
    project.idea = "主角看向镜头，说：“别回头。”"
    project.story_dir.joinpath("idea.md").write_text(project.idea)
    providers.llm = DialogueLLM()

    ClipStage().run(project, providers)

    shot = json.loads(project.path("storyboard", "shots.json").read_text())["shots"][0]
    assert shot["dialogue"] == [{"character": "主角", "line": "别回头。"}]


def test_clip_allows_validated_llm_dialogue_when_speech_has_no_quote(tmp_path):
    project, providers = _clip_project(tmp_path)
    project.idea = "Subject talks to camera"
    project.story_dir.joinpath("idea.md").write_text(project.idea)
    providers.llm = DialogueLLM()

    ClipStage().run(project, providers)

    shot = json.loads(project.path("storyboard", "shots.json").read_text())["shots"][0]
    assert shot["dialogue"] == [{"character": "Subject", "line": "Watch this."}]


# --- Language-agnostic dialogue validation (no idea-text regex; trust LLM output) ---


def test_validated_clip_dialogue_coerces_unknown_speaker_to_cast():
    from studio_agent.stages.clip import _validated_clip_dialogue
    result = _validated_clip_dialogue([{"character": "Stranger", "line": "Hi."}], ["Mara"])
    assert result == [{"character": "Mara", "line": "Hi."}]


def test_validated_clip_dialogue_drops_empty_lines_and_keeps_empty_list():
    from studio_agent.stages.clip import _validated_clip_dialogue
    assert _validated_clip_dialogue([], ["Mara"]) == []
    assert _validated_clip_dialogue([{"character": "Mara", "line": "  "}], ["Mara"]) == []


def test_validated_clip_dialogue_defaults_speaker_when_missing_and_no_cast():
    from studio_agent.stages.clip import _validated_clip_dialogue
    assert _validated_clip_dialogue([{"line": "Hi."}], []) == [
        {"character": "Speaker", "line": "Hi."}
    ]


def test_validated_clip_dialogue_preserves_chinese_line_verbatim():
    from studio_agent.stages.clip import _validated_clip_dialogue
    result = _validated_clip_dialogue([{"character": "主角", "line": "别回头。"}], ["主角"])
    assert result == [{"character": "主角", "line": "别回头。"}]


def test_clip_prompt_owns_dialogue_rules():
    # The suppression + verbatim guarantees live in the prompt (LLM honors them in any
    # language), not in idea-text regexes. Assert the prompt still instructs both.
    text = _PROMPT_TEMPLATE.lower()
    assert "return []" in text and "explicitly" in text  # only-when-requested
    assert "exactly" in text                              # preserve spoken words verbatim


# ---------------------------------------------------------------------------
# Grid-aware keyframe fixtures + tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def clip_project(tmp_path):
    """A clip-mode project that has gone through concept + bible so references exist."""
    project, _ = _clip_project(tmp_path, clip_count=1, clip_seconds=15)
    return project


@pytest.fixture()
def fake_providers():
    """Default FakeImageGen/FakeVideoGen with supports_storyboard_grid=True."""
    return Providers(
        llm=FakeLLM(),
        image=FakeImageGen(supports_storyboard_grid=True),
        video=FakeVideoGen(supports_storyboard_grid=True),
        reference_analyzer=FakeReferenceAnalyzer(),
    )


def _enable_grid(project):
    project.model_config["motion_grid"] = {"enabled": True, "layout": "2x2"}


def test_clip_renders_grid_when_capable(clip_project, fake_providers):
    # fake_providers: FakeImageGen/FakeVideoGen default supports_storyboard_grid=True
    _enable_grid(clip_project)
    ClipStage().run(clip_project, fake_providers)
    _render_approved_keyframes(clip_project, fake_providers)
    shots = json.loads(clip_project.path("storyboard", "shots.json").read_text())["shots"]
    shot = shots[0]
    assert shot["motion_grid"]["layout"] == "2x2"
    assert shot["motion_grid"]["panel_count"] == 4
    assert clip_project.path("storyboard", "keyframes", shot["keyframe"]).is_file()
    assert clip_project.path("storyboard", "prompts", f"{shot['id']}.grid.brief.md").is_file()
    assert clip_project.path("storyboard", "prompts", f"{shot['id']}.grid.md").is_file()


def test_clip_falls_back_to_single_keyframe_when_not_capable(clip_project):
    providers = Providers(
        llm=FakeLLM(),
        image=FakeImageGen(supports_storyboard_grid=False),
        video=FakeVideoGen(supports_storyboard_grid=False),
        reference_analyzer=FakeReferenceAnalyzer(),
    )
    _enable_grid(clip_project)
    ClipStage().run(clip_project, providers)
    _render_approved_keyframes(clip_project, providers)
    shots = json.loads(clip_project.path("storyboard", "shots.json").read_text())["shots"]
    assert "motion_grid" not in shots[0]
    assert not clip_project.path("storyboard", "prompts", f"{shots[0]['id']}.grid.md").is_file()
    assert clip_project.path("storyboard", "keyframes", shots[0]["keyframe"]).is_file()


def test_clip_grid_render_is_idempotent(clip_project, fake_providers):
    _enable_grid(clip_project)
    ClipStage().run(clip_project, fake_providers)
    _render_approved_keyframes(clip_project, fake_providers)
    kf = clip_project.path("storyboard", "keyframes", "sh-001.png")
    before = kf.read_bytes()
    KeyframesStage().run(clip_project, fake_providers)
    assert kf.read_bytes() == before


def test_clip_auto_sizes_clips_from_beat_complexity(tmp_path):
    # auto mode: total comes from the beats' duration_s, split into <=15s clips.
    project, providers = _clip_project(tmp_path)
    project.model_config["clip_target_duration_s"] = "auto"

    ClipStage().run(project, providers)

    seq = json.loads(project.path("storyboard", "visual_sequence.json").read_text())
    shots = json.loads(project.path("storyboard", "shots.json").read_text())["shots"]
    assert seq["target_duration_s"] == sum(
        int(round(b["duration_s"])) for b in seq["beats"]
    )
    assert all(4 <= s["duration_s"] <= 15 for s in shots)
    assert sum(s["duration_s"] for s in shots) == seq["target_duration_s"]


def test_clip_auto_tolerates_non_numeric_ceiling(tmp_path):
    # A hand-edited/non-numeric clip_max_total_s must fall back to 60s, not crash.
    project, providers = _clip_project(tmp_path)
    project.model_config["clip_target_duration_s"] = "auto"
    project.model_config["clip_max_total_s"] = "not-a-number"

    result = ClipStage().run(project, providers)

    assert result.status == "complete"
    shots = json.loads(project.path("storyboard", "shots.json").read_text())["shots"]
    assert all(4 <= s["duration_s"] <= 15 for s in shots)


class RecordingImageGen(FakeImageGen):
    """Fake image provider that records the reference_images per output file."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.refs_by_out = {}

    def generate(self, prompt, *, out_path, reference_images=None, **kwargs):
        self.refs_by_out[Path(out_path).name] = list(reference_images or [])
        return super().generate(
            prompt, out_path=out_path, reference_images=reference_images, **kwargs
        )


def test_clip_prepares_prompts_without_generating_keyframes(tmp_path):
    project, providers = _clip_project(tmp_path)
    image = RecordingImageGen()
    providers.image = image

    result = ClipStage().run(project, providers)

    assert result.status == "complete"
    assert list(project.path("storyboard", "prompts").glob("*.keyframe.md"))
    assert image.refs_by_out == {}
    assert not list(project.path("storyboard", "keyframes").glob("*.png"))


def test_clip_chains_keyframe_reference_to_previous_clip(tmp_path):
    project, providers = _clip_project(tmp_path, clip_count=2, clip_seconds=15)
    recorder = RecordingImageGen()
    providers.image = recorder

    ClipStage().run(project, providers)
    _render_approved_keyframes(project, providers)

    prev_abs = str(project.dir / "storyboard" / "keyframes" / "sh-001.png")
    # Clip 2's keyframe is conditioned on clip 1's keyframe; clip 1 is not.
    assert prev_abs in recorder.refs_by_out["sh-002.png"]
    assert prev_abs not in recorder.refs_by_out["sh-001.png"]


def test_clip_continuity_guard_in_non_first_keyframe_brief(tmp_path):
    project, providers = _clip_project(tmp_path, clip_count=2, clip_seconds=15)
    ClipStage().run(project, providers)

    guard = "Continue directly from the supplied previous frame"
    brief_1 = project.path("storyboard", "prompts", "sh-001.brief.md").read_text()
    brief_2 = project.path("storyboard", "prompts", "sh-002.brief.md").read_text()
    assert guard in brief_2
    assert guard not in brief_1
