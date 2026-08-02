import json

from studio_agent.orchestrator.project import Project
from studio_agent.prompt_approvals import require_prompt_batch_approval
from studio_agent.providers.fake import FakeImageGen, FakeLLM, FakeVideoGen
from studio_agent.stages.base import Providers
from studio_agent.stages.video_prompts import VideoPromptsStage


class RecordingVideoGen(FakeVideoGen):
    def __init__(self):
        super().__init__()
        self.calls = []

    def generate(self, prompt, *, out_path, **kwargs):
        self.calls.append({"prompt": prompt, "out_path": out_path, **kwargs})
        return super().generate(prompt, out_path=out_path, **kwargs)


def _project(tmp_path, *, shot_count=1):
    project = Project.create(
        "review motion prompts",
        root=tmp_path,
        stages=["keyframes", "video_prompts", "video"],
    )
    shots = []
    for index in range(1, shot_count + 1):
        shot_id = f"sh-{index:03d}"
        shots.append({
            "id": shot_id,
            "scene": 1,
            "description": "Mara at the workshop door",
            "camera": "medium shot",
            "camera_movement": "slow push-in",
            "action": "Mara opens the workshop door",
            "start_frame": "Mara holds the brass handle",
            "end_frame": "Mara stands inside the open doorway",
            "dialogue": [],
            "duration_s": 4.0,
            "characters": ["Mara"],
            "deps": [f"sh-{index - 1:03d}"] if index > 1 else [],
            "reference_seed": 11,
            "keyframe": f"{shot_id}.png",
        })
        FakeImageGen().generate(
            "keyframe",
            out_path=str(project.path("storyboard", "keyframes", f"{shot_id}.png")),
            seed=11,
        )
    project.path("storyboard", "shots.json").write_text(json.dumps({"shots": shots}))
    video = RecordingVideoGen()
    return project, Providers(llm=FakeLLM(), video=video), video


def test_video_prompts_writes_prompts_without_calling_video_provider(tmp_path):
    project, providers, video = _project(tmp_path)

    VideoPromptsStage().run(project, providers)

    assert project.path("storyboard", "prompts", "sh-001.video.md").is_file()
    assert video.calls == []


def test_video_prompts_describes_expected_carry_without_requiring_file(tmp_path):
    project, providers, _video = _project(tmp_path, shot_count=2)

    VideoPromptsStage().run(project, providers)

    second = project.path("storyboard", "prompts", "sh-002.video.md").read_text()
    assert "previous final frame" in second.lower()
    assert not project.path("assets", "clips", "sh-001.last_frame.png").exists()


def test_video_prompt_approval_hook_records_video_hashes(tmp_path):
    project, providers, _video = _project(tmp_path)
    stage = VideoPromptsStage()
    stage.run(project, providers)

    stage.on_approve(project, auto=False)

    require_prompt_batch_approval(project, "videos")


def test_camera_movement_skill_file_loads_in_both_languages():
    from studio_agent.runtime_skills import load_prompt_skill

    text = load_prompt_skill("camera_movement")
    assert text
    assert "one-move rule" in text.lower() or "one primary move" in text.lower()
    assert "运镜" in text  # 中文 half present (invariant #10)


def test_compiled_sound_section_humanizes_language_code(tmp_path):
    # The stage stores the language CODE ("en") in model_config, but model-facing prompt
    # text must read the human-readable LABEL ("English"), never the bare token "en".
    from studio_agent.language import language_label

    assert language_label("en") == "English"
    assert language_label("zh") == "Chinese"

    project, providers, _video = _project(tmp_path)
    # Give the shot dialogue so the Sound block emits the intelligibility direction.
    shots_path = project.path("storyboard", "shots.json")
    data = json.loads(shots_path.read_text())
    data["shots"][0]["dialogue"] = [{"character": "Mara", "line": "I shouldn't be here."}]
    shots_path.write_text(json.dumps(data))

    VideoPromptsStage().run(project, providers)

    # The compiled brief carries the deterministic Sound section verbatim (the .video.md is
    # the director's dense rewrite); this is the seam that proves the label threaded through.
    brief = project.path("storyboard", "prompts", "sh-001.video.brief.md").read_text()
    sound = brief.split("## Sound\n", 1)[1].split("\n##", 1)[0]
    assert "intelligible in English" in sound
    assert "intelligible in en" not in sound


def test_video_brief_includes_character_motion_skill(tmp_path):
    from studio_agent.runtime_skills import load_prompt_skill

    # Signature marker: a distinctive phrase from the real skill file.
    assert "follow-through" in load_prompt_skill("character_motion").lower()

    project, providers, _video = _project(tmp_path)
    VideoPromptsStage().run(project, providers)

    brief = project.path("storyboard", "prompts", "sh-001.video.brief.md").read_text()
    assert "## Character motion skill" in brief
