"""Tests for the Project model — files are the source of truth (invariant #1)."""

import json

import pytest

from studio_agent.orchestrator.project import Project, ProjectConfigurationConflict
from studio_agent.providers.base import Generation

PIPELINE = ["plot", "script", "bible"]


def test_create_builds_folder_tree_and_project_json(tmp_path):
    p = Project.create("a lonely lighthouse keeper", root=tmp_path, stages=PIPELINE)

    proj_dir = tmp_path / p.project_id
    # The PLAN.md tree exists.
    for sub in ["story", "bible", "storyboard/prompts", "storyboard/keyframes",
                "assets/clips", "assets/audio", "assets/qc", "edit", "output"]:
        assert (proj_dir / sub).is_dir(), f"missing {sub}"
    assert (proj_dir / "project.json").is_file()


def test_project_id_is_slugged_from_idea(tmp_path):
    p = Project.create("A Lonely Lighthouse Keeper!", root=tmp_path, stages=PIPELINE)
    assert p.project_id.startswith("a-lonely-lighthouse-keeper")


def test_json_round_trip_preserves_state(tmp_path):
    p = Project.create("idea one", root=tmp_path, stages=PIPELINE)
    p.set_stage_status("plot", "complete")
    p.current_stage = "script"
    p.save()

    loaded = Project.load(tmp_path / p.project_id)
    assert loaded.project_id == p.project_id
    assert loaded.current_stage == "script"
    assert loaded.stage_status("plot") == "complete"


def test_load_existing_is_idempotent_open(tmp_path):
    p = Project.create("idea two", root=tmp_path, stages=PIPELINE)
    again = Project.create("idea two", root=tmp_path, stages=PIPELINE)
    assert again.project_id == p.project_id  # reuses, does not duplicate


def test_same_slug_prefix_different_idea_creates_new_project(tmp_path):
    prefix = "a lighthouse keeper discovers a hidden city under"
    first = f"{prefix} the north pier"
    second = f"{prefix} the south pier"

    p1 = Project.create(first, root=tmp_path, stages=PIPELINE)
    p2 = Project.create(second, root=tmp_path, stages=PIPELINE)

    assert p2.project_id != p1.project_id
    assert p2.idea == second


def test_existing_project_is_extended_when_pipeline_gains_stages(tmp_path):
    p = Project.create(
        "idea grows",
        root=tmp_path,
        stages=["plot"],
        model_config={"llm": "fake"},
    )
    p.set_stage_status("plot", "approved")
    p.current_stage = None
    p.status = "done"
    p.save()

    reopened = Project.create(
        "idea grows",
        root=tmp_path,
        stages=["plot", "script", "video"],
        cost_cap=5.0,
        model_config={"llm": "fake", "video": "fake"},
    )

    assert reopened.project_id == p.project_id
    assert reopened.stages == ["plot", "script", "video"]
    assert reopened.current_stage == "script"
    assert reopened.status == "in_progress"
    assert reopened.stage_status("plot") == "approved"
    assert reopened.model_config["video"] == "fake"
    assert reopened.cost_cap == 5.0


def test_exact_idea_rejects_conflicting_model_configuration_without_mutation(tmp_path):
    first = Project.create(
        "same configured idea",
        root=tmp_path,
        stages=["plot"],
        model_config={"video": "fake", "language": "en"},
    )
    before = first.json_path.read_text()

    with pytest.raises(ProjectConfigurationConflict, match="video"):
        Project.create(
            "same configured idea",
            root=tmp_path,
            stages=["plot", "video"],
            model_config={"video": "xai", "language": "en"},
        )

    assert first.json_path.read_text() == before


def test_exact_idea_accepts_identical_and_additive_configuration(tmp_path):
    first = Project.create(
        "same additive idea",
        root=tmp_path,
        stages=["plot"],
        model_config={"video": "fake"},
    )

    reopened = Project.create(
        "same additive idea",
        root=tmp_path,
        stages=["plot", "video"],
        model_config={"video": "fake", "vlm": "fake"},
    )

    assert reopened.project_id == first.project_id
    assert reopened.model_config["vlm"] == "fake"


def test_add_cost_accumulates_into_log(tmp_path):
    p = Project.create("idea three", root=tmp_path, stages=PIPELINE)
    p.add_cost(stage="plot", provider="fake", cost_usd=0.0, seconds=0.5)
    p.add_cost(stage="plot", provider="fake", cost_usd=1.25, seconds=2.0)

    assert len(p.cost_log) == 2
    assert p.total_cost() == 1.25
    on_disk = json.loads((tmp_path / p.project_id / "project.json").read_text())
    assert len(on_disk["cost_log"]) == 2


def test_add_generation_cost_persists_tracking_without_double_counting(tmp_path):
    p = Project.create("tracked generation", root=tmp_path, stages=PIPELINE)
    gen = Generation(
        content="x",
        provider="ark",
        model="m",
        cost_usd=3.36,
        seconds=2.0,
        meta={"cost_tracking": {
            "usage": {"input_tokens": 1_000_000},
            "native_cost": 24.0,
            "native_currency": "CNY",
            "usd_conversion_rate": 0.14,
            "estimate": True,
            "pricing": {
                "model": "m",
                "source": "official",
                "as_of": "2026-07-01",
            },
        }},
    )

    p.add_generation_cost(stage="plot", generation=gen)

    assert p.total_cost() == 3.36
    assert p.cost_log[0]["native_cost"] == 24.0
    assert p.cost_log[0]["usage"]["input_tokens"] == 1_000_000
    on_disk = json.loads(p.json_path.read_text())
    assert on_disk["cost_log"][0]["native_currency"] == "CNY"


def test_unpriced_generation_is_marked_incomplete(tmp_path):
    p = Project.create("unpriced generation", root=tmp_path, stages=PIPELINE)
    gen = Generation(
        content="x", provider="remote", model="m", cost_usd=0.0
    )

    p.add_generation_cost(stage="plot", generation=gen)

    assert p.cost_log[0]["estimate"] is True
    assert p.cost_log[0]["usage_missing"] is True
    assert p.cost_log[0]["pricing"]["status"] == "unpriced"


def test_legacy_or_usage_missing_cost_entries_are_incomplete(tmp_path):
    legacy = Project.create("legacy ledger", root=tmp_path, stages=PIPELINE)
    legacy.add_cost(stage="plot", provider="remote", cost_usd=0.0, seconds=1.0)
    assert legacy.has_incomplete_cost_history() is True

    tracked = Project.create("tracked ledger", root=tmp_path, stages=PIPELINE)
    tracked.add_generation_cost(
        stage="plot",
        generation=Generation(
            content="x",
            provider="remote",
            model="m",
            cost_usd=1.0,
            meta={"cost_tracking": {
                "usage": {"input_tokens": 100},
                "native_cost": 1.0,
                "native_currency": "USD",
                "estimate": True,
                "usage_missing": False,
                "pricing": {"model": "m", "source": "official", "as_of": "now"},
            }},
        ),
    )
    assert tracked.has_incomplete_cost_history() is False


def test_generation_tracking_cannot_replace_ledger_identity_fields(tmp_path):
    p = Project.create("protected ledger", root=tmp_path, stages=PIPELINE)
    gen = Generation(
        content="x",
        provider="remote",
        model="m",
        meta={"cost_tracking": {"cost_usd": 999}},
    )

    with pytest.raises(ValueError, match="cannot replace"):
        p.add_generation_cost(stage="plot", generation=gen)


def test_cost_cap_check(tmp_path):
    p = Project.create("idea four", root=tmp_path, stages=PIPELINE, cost_cap=2.0)
    assert p.within_cost_cap() is True
    p.add_cost(stage="video", provider="fake", cost_usd=3.0, seconds=1.0)
    assert p.within_cost_cap() is False


def test_root_config_defaults_new_projects_to_unlimited():
    from studio_agent import cli

    assert cli.load_config()["cost_cap"] is None


def test_stage_path_helpers(tmp_path):
    p = Project.create("idea five", root=tmp_path, stages=PIPELINE)
    assert p.story_dir.name == "story"
    assert p.path("story", "plot.json") == p.story_dir / "plot.json"


def _old_clip_project(tmp_path, *, clip_count=2):
    stages = ["style", "concept", "bible", "clip", "video", "audio", "assemble"]
    project = Project.create("old clip", root=tmp_path, stages=stages)
    shots = []
    for index in range(1, 3):
        shot_id = f"sh-{index:03d}"
        shots.append({"id": shot_id, "keyframe": f"{shot_id}.png"})
        project.path("storyboard", "prompts", f"{shot_id}.keyframe.md").write_text(
            f"static prompt {index}"
        )
        project.path("storyboard", "prompts", f"{shot_id}.video.md").write_text(
            f"motion prompt {index}"
        )
        project.path("storyboard", "keyframes", f"{shot_id}.png").write_bytes(
            f"keyframe-{index}".encode()
        )
        if index <= clip_count:
            project.path("assets", "clips", f"{shot_id}.mp4").write_bytes(
                f"clip-{index}".encode()
            )
    project.path("storyboard", "shots.json").write_text(json.dumps({"shots": shots}))
    for stage in ("style", "concept", "bible", "clip"):
        project.set_stage_status(stage, "approved")
    project.current_stage = "video"
    project.save()
    return project


def _paid_bytes(project):
    return {
        path.relative_to(project.dir).as_posix(): path.read_bytes()
        for root in ("storyboard/keyframes", "assets/clips")
        for path in project.path(*root.split("/")).glob("*")
        if path.is_file()
    }


def test_load_migrates_old_clip_pipeline_in_order_once(tmp_path):
    project = Project.create(
        "old empty clip",
        root=tmp_path,
        stages=["style", "concept", "bible", "clip", "video", "audio", "assemble"],
    )
    project.path("storyboard", "shots.json").write_text(json.dumps({"shots": []}))
    project.save()

    first = Project.load(project.dir)
    second = Project.load(project.dir)

    expected = [
        "style", "concept", "bible", "clip", "keyframes",
        "video_prompts", "video", "audio", "assemble",
    ]
    assert first.stages == expected
    assert second.stages == expected


def test_migration_preserves_existing_paid_artifact_bytes(tmp_path):
    project = _old_clip_project(tmp_path, clip_count=2)
    before = _paid_bytes(project)

    migrated = Project.load(project.dir)

    assert _paid_bytes(migrated) == before
    assert migrated.stage_status("keyframes") == "approved"
    assert migrated.stage_status("video_prompts") == "approved"


def test_partial_old_video_returns_to_video_prompt_review(tmp_path):
    project = _old_clip_project(tmp_path, clip_count=1)

    migrated = Project.load(project.dir)

    assert migrated.current_stage == "video_prompts"
    assert migrated.stage_status("video_prompts") == "complete"
