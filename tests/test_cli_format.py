"""CLI tests for product format and language selection."""

import json

from studio_agent import cli
from studio_agent.orchestrator.project import Project
from studio_agent.providers.fake import FakeImageGen, FakeLLM, FakeVideoGen, FakeVLMCheck
from studio_agent.stages.base import Providers
from studio_agent.stages.plot import PlotStage
from studio_agent.stages.script import ScriptStage
from studio_agent.stages.storyboard import StoryboardStage


def _only_project(root):
    projects = list(root.iterdir())
    assert len(projects) == 1
    return Project.load(projects[0])


def test_new_projects_default_to_native_video_audio():
    assert cli.new_project_audio_mode({"video": "fake"}) == "native_video"
    assert cli.new_project_audio_mode({"audio_mode": "legacy"}) == "legacy"


def test_fake_providers_support_native_audio_and_review():
    assert FakeVideoGen().capabilities.supports_native_audio is True
    assert FakeVLMCheck().supports_audio_review is True


def test_seedance_2_and_1_5_enable_audio_but_1_0_does_not():
    videos = cli.load_config()["model_options"]["video"]
    assert videos["fal-seedance"]["video_generate_audio"] is True
    assert videos["china-seedance-fast"]["video_generate_audio"] is True
    assert videos["china-seedance-1-5-pro"]["video_generate_audio"] is True
    assert videos["china-seedance-1-0-pro"]["video_generate_audio"] is False


def test_run_format_preset_is_stored_in_project_config(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "PROJECTS_ROOT", tmp_path)

    code = cli.main([
        "run",
        "--idea", "a secret elevator opens under a noodle shop",
        "--fake",
        "--format", "short_film",
        "--language", "en",
    ])

    assert code == 0
    capsys.readouterr()
    project = _only_project(tmp_path)
    assert project.model_config["format_name"] == "short_film"
    assert project.model_config["product_format"]["min_duration_s"] == 120
    assert project.model_config["language"] == "en"
    assert project.model_config["audio_mode"] == "native_video"


def test_film_duration_minutes_overrides_story_duration_bounds(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "PROJECTS_ROOT", tmp_path)

    code = cli.main([
        "run",
        "--idea", "a lighthouse keeper hears a warning",
        "--fake",
        "--format", "short_film",
        "--duration-min", "2",
    ])

    assert code == 0
    capsys.readouterr()
    project = _only_project(tmp_path)
    fmt = project.model_config["product_format"]
    assert fmt["target_duration_s"] == 120
    assert fmt["min_duration_s"] == 84
    assert fmt["max_duration_s"] == 168


def test_motion_grid_config_is_carried_into_project_model_config(tmp_path, monkeypatch, capsys):
    """motion_grid from config.yaml must appear in project.model_config so the grid gate works."""
    monkeypatch.setattr(cli, "PROJECTS_ROOT", tmp_path)

    code = cli.main([
        "run",
        "--idea", "a secret elevator opens under a noodle shop",
        "--fake",
        "--language", "en",
    ])

    assert code == 0
    capsys.readouterr()
    project = _only_project(tmp_path)
    assert "motion_grid" in project.model_config
    assert project.model_config["motion_grid"] == {"enabled": False, "story_scenes": False, "layout": "auto"}


def test_short_video_clip_duration_choice_is_persisted(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "PROJECTS_ROOT", tmp_path)

    code = cli.main([
        "run",
        "--idea", "a man transforms into a werewolf",
        "--fake",
        "--format", "short_video",
        "--clip-duration", "30",
    ])

    assert code == 0
    capsys.readouterr()
    project = _only_project(tmp_path)
    assert project.model_config["clip_target_duration_s"] == 30


def test_story_formats_ignore_clip_duration_choice(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "PROJECTS_ROOT", tmp_path)

    code = cli.main([
        "run",
        "--idea", "a lighthouse keeper hears a warning",
        "--fake",
        "--format", "short_film",
        "--clip-duration", "60",
    ])

    assert code == 0
    capsys.readouterr()
    project = _only_project(tmp_path)
    assert "clip_target_duration_s" not in project.model_config


def test_run_unknown_format_fails_before_creating_project(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "PROJECTS_ROOT", tmp_path)

    code = cli.main([
        "run",
        "--idea", "a secret elevator opens under a noodle shop",
        "--fake",
        "--format", "feature_film",
    ])

    out = capsys.readouterr().out
    assert code == 1
    assert "unknown format 'feature_film'" in out
    assert "short_film" in out
    assert list(tmp_path.iterdir()) == []


def test_short_drama_fake_storyboard_targets_60_to_90_seconds(tmp_path):
    p = Project.create(
        "a secret elevator opens under a noodle shop",
        root=tmp_path,
        stages=["plot", "script", "storyboard"],
        model_config={
            "language": "en",
            "format_name": "short_drama_episode",
            "product_format": {
                "name": "short_drama_episode",
                "target_duration_s": 75,
                "min_duration_s": 60,
                "max_duration_s": 90,
            },
        },
    )
    p.story_dir.joinpath("idea.md").write_text("a secret elevator opens under a noodle shop")
    providers = Providers(llm=FakeLLM(), image=FakeImageGen())

    PlotStage().run(p, providers)
    ScriptStage().run(p, providers)
    StoryboardStage().run(p, providers)

    shots = json.loads(p.path("storyboard", "shots.json").read_text())["shots"]
    duration = sum(s["duration_s"] for s in shots)
    assert 60 <= duration <= 90


def test_chinese_language_is_preserved_by_fake_story_stages(tmp_path):
    p = Project.create(
        "一个灯塔守望者遇见会说话的海鸥",
        root=tmp_path,
        stages=["plot", "script"],
        model_config={"language": "zh"},
    )
    p.story_dir.joinpath("idea.md").write_text("一个灯塔守望者遇见会说话的海鸥")
    providers = Providers(llm=FakeLLM())

    PlotStage().run(p, providers)
    ScriptStage().run(p, providers)

    plot = json.loads(p.path("story", "plot.json").read_text())
    script = json.loads(p.path("story", "script.json").read_text())
    assert "一部关于" in plot["logline"]
    line = script["episodes"][0]["scenes"][0]["dialogue"][0]["line"]
    assert "我们必须" in line


def test_new_project_music_defaults_off_and_flags_override():
    import argparse
    from studio_agent import cli

    base = argparse.Namespace(music=None, music_mood=None)
    assert cli.new_project_music(base, {}) == {"music_enabled": False, "music_mood": ""}

    on = argparse.Namespace(music=True, music_mood="warm melancholic piano")
    assert cli.new_project_music(on, {}) == {
        "music_enabled": True, "music_mood": "warm melancholic piano"
    }

    off = argparse.Namespace(music=False, music_mood=None)
    assert cli.new_project_music(off, {"music_enabled": True}) == {
        "music_enabled": False, "music_mood": ""
    }
