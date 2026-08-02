import json

import pytest

from studio_agent.orchestrator.project import Project
from studio_agent.prompt_approvals import PromptApprovalError, confirm_prompt_batch
from studio_agent.providers.fake import FakeImageGen, FakeLLM, FakeVideoGen
from studio_agent.stages.base import Providers
from studio_agent.stages.keyframes import KeyframesStage


class RecordingImageGen(FakeImageGen):
    def __init__(self):
        super().__init__(supports_storyboard_grid=True)
        self.calls = []

    def generate(self, prompt, *, out_path, reference_images=None, **kwargs):
        self.calls.append({
            "prompt": prompt,
            "out_path": out_path,
            "reference_images": list(reference_images or []),
        })
        return super().generate(
            prompt,
            out_path=out_path,
            reference_images=reference_images,
            **kwargs,
        )


def _project(tmp_path, *, grid=False):
    project = Project.create(
        "keyframes stage",
        root=tmp_path,
        stages=["clip", "keyframes"],
        model_config={
            "style": {},
            "motion_grid": {"enabled": grid, "layout": "2x2"},
        },
    )
    shot = {
        "id": "sh-001",
        "scene": 1,
        "keyframe": "sh-001.png",
        "reference_seed": 42,
        "camera": "50mm medium shot",
        "camera_movement": "",
        "start_frame": "Mara holds a still opening pose beside the desk",
        "description": "Quiet room at night",
        "characters": [],
        "reference_images": [],
    }
    prompt_name = "sh-001.keyframe.md"
    prompt = "50mm medium shot. Mara holds a still opening pose beside the desk."
    if grid:
        shot["motion_grid"] = {
            "layout": "2x2",
            "rows": 2,
            "cols": 2,
            "panel_count": 4,
            "panel_beats": ["opening", "middle-a", "middle-b", "final"],
        }
        prompt_name = "sh-001.grid.md"
        prompt = (
            "Grid 2x2. Panel 1: frozen wide still beside the desk. "
            "Panel 2: frozen medium still beside the monitor. "
            "Panel 3: frozen close still beside the screen. "
            "Panel 4: frozen full-body still beside the chair."
        )
    project.path("storyboard", "shots.json").write_text(
        json.dumps({"shots": [shot]})
    )
    project.path("storyboard", "prompts", prompt_name).write_text(prompt)
    image = RecordingImageGen()
    providers = Providers(
        llm=FakeLLM(),
        image=image,
        video=FakeVideoGen(supports_storyboard_grid=True),
    )
    return project, providers, image


def test_keyframes_requires_approved_prompt_batch(tmp_path):
    project, providers, image = _project(tmp_path)

    with pytest.raises(PromptApprovalError):
        KeyframesStage().run(project, providers)

    assert image.calls == []


def test_keyframes_generates_all_approved_missing_images(tmp_path):
    project, providers, image = _project(tmp_path)
    confirm_prompt_batch(project, "keyframes", confirmer="human")

    result = KeyframesStage().run(project, providers)

    assert result.status == "complete"
    assert project.path("storyboard", "keyframes", "sh-001.png").is_file()
    assert len(image.calls) == 1
    assert project.cost_log[-1]["stage"] == "keyframes"


def test_keyframes_uses_grid_prompt_in_explicit_capable_grid_mode(tmp_path):
    project, providers, image = _project(tmp_path, grid=True)
    confirm_prompt_batch(project, "keyframes", confirmer="human")

    KeyframesStage().run(project, providers)

    expected = project.path("storyboard", "prompts", "sh-001.grid.md").read_text()
    assert image.calls[0]["prompt"] == expected


def test_keyframes_skips_existing_approved_image(tmp_path):
    project, providers, image = _project(tmp_path)
    confirm_prompt_batch(project, "keyframes", confirmer="human")
    FakeImageGen().generate(
        "existing",
        out_path=str(project.path("storyboard", "keyframes", "sh-001.png")),
        seed=42,
    )

    KeyframesStage().run(project, providers)

    assert image.calls == []
