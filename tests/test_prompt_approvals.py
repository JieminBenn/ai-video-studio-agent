import json

import pytest

from studio_agent.orchestrator.project import Project
from studio_agent.prompt_approvals import (
    PromptApprovalError,
    confirm_prompt,
    confirm_prompt_batch,
    invalidate_prompt_approvals,
    require_prompt_batch_approval,
)


def _project(tmp_path):
    project = Project.create(
        "approval test",
        root=tmp_path,
        stages=["clip", "keyframes"],
        model_config={
            "image_model": "model-a",
            "image_max_reference_images": 3,
        },
    )
    project.path("storyboard", "shots.json").write_text(json.dumps({"shots": [
        {"id": "sh-001", "keyframe": "sh-001.png", "camera_movement": ""},
        {"id": "sh-002", "keyframe": "sh-002.png", "camera_movement": ""},
    ]}))
    for shot_id in ("sh-001", "sh-002"):
        project.path("storyboard", "prompts", f"{shot_id}.keyframe.md").write_text(
            f"Static 50mm opening frame for {shot_id}."
        )
    return project


def _manifest(project):
    return json.loads(
        project.path(
            "storyboard", "prompt_approvals", "keyframes.json"
        ).read_text()
    )


def test_confirm_batch_records_exact_sha256_and_requires_unchanged_files(tmp_path):
    project = _project(tmp_path)

    manifest = confirm_prompt_batch(project, "keyframes", confirmer="human")

    assert set(manifest["prompts"]) == {"sh-001", "sh-002"}
    assert len(manifest["prompts"]["sh-001"]["sha256"]) == 64
    require_prompt_batch_approval(project, "keyframes")

    project.path("storyboard", "prompts", "sh-001.keyframe.md").write_text(
        "Changed static still."
    )
    with pytest.raises(PromptApprovalError, match="sh-001"):
        require_prompt_batch_approval(project, "keyframes")


def test_selected_confirmation_updates_only_selected_hash(tmp_path):
    project = _project(tmp_path)
    confirm_prompt_batch(project, "keyframes", confirmer="human")
    before = _manifest(project)
    project.path("storyboard", "prompts", "sh-001.keyframe.md").write_text(
        "Revised static still."
    )

    confirm_prompt(project, "keyframes", "sh-001", confirmer="human")

    after = _manifest(project)
    assert after["prompts"]["sh-002"] == before["prompts"]["sh-002"]
    assert after["prompts"]["sh-001"] != before["prompts"]["sh-001"]


def test_invalidate_selected_approval_preserves_unrelated_shot(tmp_path):
    project = _project(tmp_path)
    confirm_prompt_batch(project, "keyframes", confirmer="human")

    invalidate_prompt_approvals(project, "keyframes", shot_ids=["sh-001"])

    manifest = _manifest(project)
    assert "sh-001" not in manifest["prompts"]
    assert "sh-002" in manifest["prompts"]


def test_provider_context_change_invalidates_batch(tmp_path):
    project = _project(tmp_path)
    confirm_prompt_batch(project, "keyframes", confirmer="human")
    project.model_config["image_max_reference_images"] = 1
    project.save()

    with pytest.raises(PromptApprovalError, match="generation context"):
        require_prompt_batch_approval(project, "keyframes")


def test_motion_grid_approval_hashes_grid_prompt_instead_of_keyframe_prompt(tmp_path):
    project = _project(tmp_path)
    shots = json.loads(project.path("storyboard", "shots.json").read_text())
    shots["shots"] = [{
        "id": "sh-001",
        "keyframe": "sh-001.png",
        "camera_movement": "",
        "motion_grid": {"panel_count": 2, "rows": 1, "cols": 2},
    }]
    project.path("storyboard", "shots.json").write_text(json.dumps(shots))
    project.path("storyboard", "prompts", "sh-001.grid.md").write_text(
        "Grid 1x2. Panel 1: frozen wide still. Panel 2: frozen close still."
    )

    manifest = confirm_prompt_batch(project, "keyframes", confirmer="human")

    assert manifest["prompts"]["sh-001"]["path"].endswith("sh-001.grid.md")
    require_prompt_batch_approval(project, "keyframes")
