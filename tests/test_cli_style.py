"""CLI tests for strict named style presets."""

import json

from studio_agent import cli
from studio_agent.orchestrator.project import Project
from studio_agent.style import format_style_prompt, resolve_style


def _only_project(root):
    projects = list(root.iterdir())
    assert len(projects) == 1
    return Project.load(projects[0])


def test_3d_preset_resolves_to_general_3d_visual_prompt():
    resolved = resolve_style(cli.load_config(), "3d")
    prompt = format_style_prompt(resolved.style).lower()

    assert resolved.name == "3d"
    for trait in ("3d", "pbr", "volumetric", "hair and cloth"):
        assert trait in prompt
    # Generalized: the preset is no longer locked to Chinese donghua/xianxia.
    for locked in ("donghua", "xianxia", "chinese", "guoman"):
        assert locked not in prompt


def test_style_prompt_excludes_director_only_playbook():
    # The per-style prompt_playbook is consumed by the image-prompt director, not the
    # compact style sentence used in briefs / video prompts / bible style.md.
    from studio_agent.style import format_style_markdown

    style = {"look": "cinematic", "palette": "natural", "prompt_playbook": "SECRET PLAYBOOK"}
    assert "SECRET PLAYBOOK" not in format_style_prompt(style)
    assert "Prompt Playbook" not in format_style_prompt(style)
    assert "SECRET PLAYBOOK" not in format_style_markdown(style)


def test_run_style_preset_is_stored_in_project_config(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "PROJECTS_ROOT", tmp_path)

    code = cli.main([
        "run",
        "--idea", "a baker finds a comet in the oven",
        "--fake",
        "--style", "anime",
    ])

    assert code == 0
    capsys.readouterr()
    project = _only_project(tmp_path)
    assert project.model_config["style_name"] == "anime"
    assert project.model_config["style"]["look"] == "anime"
    assert "cel" in json.dumps(project.model_config["style"]).lower()


def test_run_genre_is_locked_into_project_config(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "PROJECTS_ROOT", tmp_path)

    code = cli.main([
        "run",
        "--idea", "白鹤仙子的传说",
        "--fake",
        "--genre", "中国古代神话",
    ])

    assert code == 0
    capsys.readouterr()
    project = _only_project(tmp_path)
    assert project.model_config["genre"] == "中国古代神话"


def test_run_without_genre_defaults_to_empty(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "PROJECTS_ROOT", tmp_path)

    code = cli.main(["run", "--idea", "a baker finds a comet", "--fake"])

    assert code == 0
    capsys.readouterr()
    project = _only_project(tmp_path)
    assert project.model_config.get("genre", "") == ""


def test_run_unknown_style_fails_before_creating_project(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "PROJECTS_ROOT", tmp_path)

    code = cli.main([
        "run",
        "--idea", "a baker finds a comet in the oven",
        "--fake",
        "--style", "watercolor-noir",
    ])

    out = capsys.readouterr().out
    assert code == 1
    assert "unknown style 'watercolor-noir'" in out
    assert "anime" in out and "cartoon" in out and "cinematic" in out
    assert list(tmp_path.iterdir()) == []
