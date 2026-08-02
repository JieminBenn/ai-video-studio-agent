"""Custom (user-defined) style resolution — the free-form path that replaces presets.

A custom style is not looked up in ``style_presets``; the style stage profiles the
user's text/image into the dict later. ``resolve_style`` must accept the custom
sentinel without raising, returning an empty placeholder the stage fills in.
"""

from studio_agent import cli
from studio_agent.style import CUSTOM_STYLE_NAME, ResolvedStyle, resolve_style


def test_custom_style_resolves_to_an_empty_placeholder():
    resolved = resolve_style(cli.load_config(), CUSTOM_STYLE_NAME)

    assert isinstance(resolved, ResolvedStyle)
    assert resolved.name == CUSTOM_STYLE_NAME
    # No preset lookup — the style stage profiles the real dict later.
    assert resolved.style == {}


def test_named_presets_still_resolve_normally():
    resolved = resolve_style(cli.load_config(), "anime")
    assert resolved.name == "anime"
    assert resolved.style["look"] == "anime"


def _only_project(root):
    from studio_agent.orchestrator.project import Project

    projects = list(root.iterdir())
    assert len(projects) == 1
    return Project.load(projects[0])


def test_run_style_description_locks_a_custom_style(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "PROJECTS_ROOT", tmp_path)

    code = cli.main([
        "run",
        "--idea", "a baker finds a comet in the oven",
        "--fake",
        "--auto",
        "--style-description", "1970s grainy analog sci-fi",
    ])

    assert code == 0
    capsys.readouterr()
    project = _only_project(tmp_path)
    assert project.model_config["style_name"] == CUSTOM_STYLE_NAME
    # The style stage profiled the description into the project style.
    import json
    assert "sci-fi" in json.dumps(project.model_config["style"]).lower()
    assert "sci-fi" in project.path("bible", "style.md").read_text().lower()


def test_run_style_image_is_saved_as_a_style_reference(tmp_path, monkeypatch, capsys):
    from studio_agent.providers.fake import FakeImageGen
    from studio_agent.reference_assets import style_reference_paths

    monkeypatch.setattr(cli, "PROJECTS_ROOT", tmp_path / "projects")
    src = tmp_path / "src"
    src.mkdir()
    img = src / "moody-noir-style.png"
    FakeImageGen().generate("style", out_path=str(img), seed=3)

    code = cli.main([
        "run",
        "--idea", "a detective in the rain",
        "--fake",
        "--auto",
        "--style-image", str(img),
    ])

    assert code == 0
    capsys.readouterr()
    project = _only_project(tmp_path / "projects")
    assert project.model_config["style_name"] == CUSTOM_STYLE_NAME
    assert style_reference_paths(project)  # the image was saved as a style/global ref


def test_run_missing_style_image_fails_before_creating_project(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "PROJECTS_ROOT", tmp_path)

    code = cli.main([
        "run",
        "--idea", "a detective in the rain",
        "--fake",
        "--style-image", str(tmp_path / "nope.png"),
    ])

    out = capsys.readouterr().out
    assert code == 1
    assert "style image not found" in out
    assert list(tmp_path.iterdir()) == []
