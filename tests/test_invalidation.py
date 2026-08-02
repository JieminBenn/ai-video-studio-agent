"""Archive-backed dependency invalidation for localized regeneration."""

import json

import pytest

from studio_agent.invalidation import (
    archive_paths,
    invalidate_artifact_revision,
    invalidate_bible_asset,
    invalidate_bible_text,
    invalidate_reference_change,
)
from studio_agent import invalidation
from studio_agent.orchestrator.project import Project
from studio_agent.prompt_approvals import confirm_prompt_batch


PIPELINE = [
    "plot", "script", "bible", "storyboard", "keyframes", "video_prompts",
    "video", "review", "audio", "assemble",
]


def _project(tmp_path):
    project = Project.create("safe invalidation", root=tmp_path, stages=PIPELINE)
    shots = {
        "shots": [
            {
                "id": "sh-001",
                "keyframe": "sh-001.png",
                "characters": ["Mara"],
                "reference_characters": ["Mara"],
                "reference_locations": ["Apartment"],
                "reference_images": ["bible/characters/mara/reference.png"],
            },
            {
                "id": "sh-002",
                "keyframe": "sh-002.png",
                "characters": ["Theo"],
                "reference_characters": ["Theo"],
                "reference_locations": ["Street"],
                "reference_images": ["bible/characters/theo/reference.png"],
            },
        ]
    }
    project.path("storyboard", "shots.json").write_text(json.dumps(shots))
    for shot_id in ("sh-001", "sh-002"):
        project.path("storyboard", "keyframes", f"{shot_id}.png").write_bytes(
            f"keyframe-{shot_id}".encode()
        )
        for suffix in ("keyframe.md", "video.md", "audio.md"):
            project.path("storyboard", "prompts", f"{shot_id}.{suffix}").write_text(
                f"prompt-{shot_id}-{suffix}"
            )
        project.path("assets", "clips", f"{shot_id}.mp4").write_bytes(
            f"clip-{shot_id}".encode()
        )
        project.path("assets", "clips", f"{shot_id}.last_frame.png").write_bytes(
            f"last-{shot_id}".encode()
        )
        project.path("assets", "qc", f"{shot_id}.json").write_text("{}")
        project.path("assets", "audio", f"{shot_id}.dialogue.wav").write_bytes(
            f"audio-{shot_id}".encode()
        )
    project.path("assets", "clips", "clips.json").write_text("{}")
    project.path("assets", "qc", "summary.json").write_text("{}")
    project.path("edit", "timeline.json").write_text("{}")
    project.path("output", f"{project.project_id}.mp4").write_bytes(b"final")

    for slug in ("mara", "theo"):
        base = project.path("bible", "characters", slug)
        base.mkdir(parents=True, exist_ok=True)
        base.joinpath("reference.png").write_bytes(f"reference-{slug}".encode())
    confirm_prompt_batch(project, "keyframes", confirmer="test")
    confirm_prompt_batch(project, "videos", confirmer="test")
    return project


def _archived(project, result, rel_path):
    return project.dir / result.archive_dir / rel_path


def test_archive_paths_moves_files_and_writes_manifest(tmp_path):
    project = _project(tmp_path)
    source = project.path("assets", "clips", "sh-001.mp4")

    result = archive_paths(
        project,
        ["assets/clips/sh-001.mp4"],
        reason="video-prompt-revised",
        source_path="storyboard/prompts/sh-001.video.md",
        affected_shot_ids=["sh-001"],
    )

    assert not source.exists()
    assert result.archived_paths == ["assets/clips/sh-001.mp4"]
    assert _archived(project, result, "assets/clips/sh-001.mp4").read_bytes() == b"clip-sh-001"
    manifest = json.loads((project.dir / result.archive_dir / "manifest.json").read_text())
    assert manifest["reason"] == "video-prompt-revised"
    assert manifest["source_path"] == "storyboard/prompts/sh-001.video.md"
    assert manifest["affected_shot_ids"] == ["sh-001"]


def test_video_prompt_revision_preserves_outputs_and_removes_only_selected_approval(tmp_path):
    project = _project(tmp_path)

    result = invalidate_artifact_revision(
        project, "storyboard/prompts/sh-001.video.md"
    )

    assert result.affected_shot_ids == ["sh-001"]
    assert project.path("assets", "clips", "sh-001.mp4").is_file()
    assert project.path("assets", "clips", "sh-002.mp4").is_file()
    assert project.path("storyboard", "prompts", "sh-001.video.md").is_file()
    assert result.archived_paths == []
    approvals = json.loads(
        project.path("storyboard", "prompt_approvals", "videos.json").read_text()
    )["prompts"]
    assert "sh-001" not in approvals
    assert "sh-002" in approvals
    assert result.invalidated_stages[0] == "video_prompts"
    assert project.stage_status("video") == "pending"
    assert project.current_stage == "video_prompts"


def test_keyframe_prompt_revision_starts_at_keyframes_and_preserves_other_approvals(tmp_path):
    project = _project(tmp_path)

    result = invalidate_artifact_revision(
        project, "storyboard/prompts/sh-001.keyframe.md"
    )

    assert result.invalidated_stages[0] == "keyframes"
    assert project.current_stage == "keyframes"
    assert not project.path("storyboard", "keyframes", "sh-001.png").exists()
    assert project.path("storyboard", "keyframes", "sh-002.png").is_file()
    keyframe_approvals = json.loads(
        project.path("storyboard", "prompt_approvals", "keyframes.json").read_text()
    )["prompts"]
    video_approvals = json.loads(
        project.path("storyboard", "prompt_approvals", "videos.json").read_text()
    )["prompts"]
    assert set(keyframe_approvals) == {"sh-002"}
    assert set(video_approvals) == {"sh-002"}


def test_video_prompt_revision_preserves_native_audio_until_regeneration(tmp_path):
    project = _project(tmp_path)
    project.model_config["audio_mode"] = "native_video"
    project.path("assets", "clips", "sh-001.native.source.wav").write_bytes(b"source-one")
    project.path("assets", "clips", "sh-002.native.source.wav").write_bytes(b"source-two")
    project.path("assets", "audio", "sh-001.native.wav").write_bytes(b"native-one")
    project.path("assets", "audio", "sh-002.native.wav").write_bytes(b"native-two")
    project.path("assets", "audio", "audio.json").write_text("{}")

    result = invalidate_artifact_revision(
        project, "storyboard/prompts/sh-001.video.md"
    )

    for rel in (
        "assets/clips/sh-001.mp4",
        "assets/clips/sh-001.last_frame.png",
        "assets/clips/sh-001.native.source.wav",
        "assets/audio/sh-001.native.wav",
        "assets/qc/sh-001.json",
        "assets/audio/audio.json",
    ):
        assert project.path(*rel.split("/")).is_file()
    for rel in (
        "assets/clips/sh-002.mp4",
        "assets/clips/sh-002.last_frame.png",
        "assets/clips/sh-002.native.source.wav",
        "assets/audio/sh-002.native.wav",
        "assets/qc/sh-002.json",
    ):
        assert project.path(*rel.split("/")).is_file()
    assert result.archived == 0


def test_video_prompt_revision_preserves_shot_narration_voiceover(tmp_path):
    project = _project(tmp_path)
    project.model_config["audio_mode"] = "native_video"
    project.path("assets", "audio", "sh-001.narration.wav").write_bytes(b"vo-one")
    project.path("assets", "audio", "sh-002.narration.wav").write_bytes(b"vo-two")

    result = invalidate_artifact_revision(
        project, "storyboard/prompts/sh-001.video.md"
    )

    assert project.path("assets", "audio", "sh-001.narration.wav").is_file()
    assert project.path("assets", "audio", "sh-002.narration.wav").is_file()
    assert result.archived == 0


def test_video_prompt_revision_preserves_manifest_declared_native_source(tmp_path):
    project = _project(tmp_path)
    project.model_config["audio_mode"] = "native_video"
    custom = project.path("assets", "clips", "provider-audio.wav")
    custom.write_bytes(b"provider-source")
    project.path("assets", "clips", "clips.json").write_text(json.dumps({
        "clips": [
            {"id": "sh-001", "clip": "sh-001.mp4", "native_audio": custom.name},
            {"id": "sh-002", "clip": "sh-002.mp4"},
        ]
    }))

    result = invalidate_artifact_revision(
        project, "storyboard/prompts/sh-001.video.md"
    )

    assert custom.read_bytes() == b"provider-source"
    assert result.archived == 0


def test_legacy_video_prompt_revision_keeps_native_named_compatibility_files(tmp_path):
    project = _project(tmp_path)
    project.model_config["audio_mode"] = "legacy"
    source = project.path("assets", "clips", "sh-001.native.source.wav")
    native = project.path("assets", "audio", "sh-001.native.wav")
    source.write_bytes(b"compat-source")
    native.write_bytes(b"compat-native")

    invalidate_artifact_revision(project, "storyboard/prompts/sh-001.video.md")

    assert source.is_file()
    assert native.is_file()


def test_native_project_ignores_obsolete_legacy_audio_prompt_revision(tmp_path):
    project = _project(tmp_path)
    project.model_config["audio_mode"] = "native_video"
    for stage in PIPELINE:
        project.set_stage_status(stage, "approved")
    project.current_stage = None
    project.status = "done"
    project.save()

    result = invalidate_artifact_revision(
        project, "storyboard/prompts/sh-001.audio.md"
    )

    assert result.invalidated_stages == []
    assert project.current_stage is None
    assert project.status == "done"
    assert all(project.stage_status(stage) == "approved" for stage in PIPELINE)


def test_native_storyboard_revision_preserves_obsolete_audio_prompts(tmp_path):
    project = _project(tmp_path)
    project.model_config["audio_mode"] = "native_video"
    obsolete = project.path("storyboard", "prompts", "sh-001.audio.md")

    invalidate_artifact_revision(project, "storyboard/shots.json")

    assert obsolete.is_file()


@pytest.mark.parametrize(
    ("prompt_kind", "audio_mode"),
    [("keyframe", "legacy"), ("video", "native_video"), ("audio", "legacy")],
)
def test_deleted_shot_prompt_revision_is_a_noop(
    tmp_path, prompt_kind, audio_mode
):
    project = _project(tmp_path)
    project.model_config["audio_mode"] = audio_mode
    for stage in PIPELINE:
        project.set_stage_status(stage, "approved")
    project.current_stage = None
    project.status = "done"
    stale = project.path(
        "storyboard", "prompts", f"deleted-shot.{prompt_kind}.md"
    )
    stale.write_text("stale")
    protected = {
        rel: project.path(*rel.split("/")).read_bytes()
        for rel in (
            "assets/clips/clips.json",
            "assets/qc/summary.json",
            "edit/timeline.json",
            f"output/{project.project_id}.mp4",
        )
    }
    project.save()

    result = invalidate_artifact_revision(
        project, f"storyboard/prompts/deleted-shot.{prompt_kind}.md"
    )

    assert result.invalidated_stages == []
    assert result.archived_paths == []
    assert project.current_stage is None
    assert project.status == "done"
    assert all(project.stage_status(stage) == "approved" for stage in PIPELINE)
    assert stale.is_file()
    for rel, content in protected.items():
        assert project.path(*rel.split("/")).read_bytes() == content


def test_music_prompt_revision_archives_cached_music_and_final_edit(tmp_path):
    project = _project(tmp_path)
    prompt = project.path("storyboard", "prompts", "music.audio.md")
    prompt.write_text("new music direction")
    music = project.path("assets", "audio", "music.wav")
    music.write_bytes(b"old music")

    result = invalidate_artifact_revision(project, "storyboard/prompts/music.audio.md")

    assert prompt.is_file()
    assert not music.exists()
    assert _archived(project, result, "assets/audio/music.wav").read_bytes() == b"old music"
    assert not project.path("edit", "timeline.json").exists()
    assert project.current_stage == "audio"


def test_matching_targets_cjk_character_by_stable_slug(tmp_path):
    """Invalidation must target a character's shots by the stable folder slug.

    Regression guard for the staleness class: a CJK-named character (青年) whose shot has not
    (yet) been re-wired with the bible reference path would match neither by ASCII name-slug
    (→ "unknown") nor by path. Matching then returned [] and fell back to invalidating EVERY
    shot — over-broad, and it hid the real gap. Joining by ``char_slug`` targets exactly the
    featuring shot even before the reference path is wired, so a regenerated character reliably
    (and only) re-renders the shots that use it.
    """
    from studio_agent.stages.bible import char_slug

    project = Project.create("cjk target", root=tmp_path, stages=PIPELINE)
    slug = char_slug("青年")
    shots = {"shots": [
        {"id": "sh-001", "keyframe": "sh-001.png", "characters": ["青年"],
         "reference_characters": ["青年"],
         "reference_images": ["bible/style_sample.png"]},  # bible char path NOT wired (legacy)
        {"id": "sh-002", "keyframe": "sh-002.png", "characters": ["Theo"],
         "reference_characters": ["Theo"],
         "reference_images": ["bible/characters/theo/reference.png"]},
    ]}
    project.path("storyboard", "shots.json").write_text(json.dumps(shots))

    matched = invalidation._matching_shot_ids(project, "character", slug)

    assert matched == ["sh-001"], "CJK character must be targeted by stable slug, not fallback-to-all"


def test_character_reference_regeneration_cascades_to_sheets_and_affected_shots(tmp_path):
    project = _project(tmp_path)

    result = invalidate_bible_asset(
        project, "bible/characters/mara/reference.png"
    )

    assert result.affected_shot_ids == ["sh-001"]
    for rel in (
        "bible/characters/mara/reference.png",
        "storyboard/keyframes/sh-001.png",
        "assets/clips/sh-001.mp4",
    ):
        assert not project.path(*rel.split("/")).exists()
        assert _archived(project, result, rel).is_file()
    # The removed combined-sheet files are no longer archived
    assert "bible/characters/mara/turnaround.png" not in result.archived_paths
    assert "bible/characters/mara/expressions.png" not in result.archived_paths
    assert project.path("bible", "characters", "theo", "reference.png").is_file()
    assert project.path("assets", "clips", "sh-002.mp4").is_file()
    assert project.stage_status("bible") == "pending"
    assert project.current_stage == "bible"


def test_character_reference_regeneration_archives_provider_prompt_sidecars(tmp_path):
    project = _project(tmp_path)
    base = project.path("bible", "characters", "mara", "reference")
    prompt_sidecar = base.with_suffix(".provider-prompt.md")
    metadata_sidecar = base.with_suffix(".provider-prompt.json")
    prompt_sidecar.write_text("exact fitted prompt")
    metadata_sidecar.write_text('{"limit": 8000}')

    result = invalidate_bible_asset(
        project, "bible/characters/mara/reference.png"
    )

    for rel in (
        "bible/characters/mara/reference.provider-prompt.md",
        "bible/characters/mara/reference.provider-prompt.json",
    ):
        assert rel in result.archived_paths
        assert not project.path(*rel.split("/")).exists()
        assert _archived(project, result, rel).is_file()


def test_keyframe_invalidation_archives_provider_prompt_sidecars(tmp_path):
    project = _project(tmp_path)
    keyframe_base = project.path("storyboard", "keyframes", "sh-001")
    keyframe_base.with_suffix(".provider-prompt.md").write_text("fitted keyframe")
    keyframe_base.with_suffix(".provider-prompt.json").write_text('{"limit": 8000}')

    result = invalidate_artifact_revision(
        project, "storyboard/prompts/sh-001.keyframe.md"
    )

    assert "storyboard/keyframes/sh-001.provider-prompt.md" in result.archived_paths
    assert "storyboard/keyframes/sh-001.provider-prompt.json" in result.archived_paths


def test_bible_source_revision_archives_provider_prompt_sidecars(tmp_path):
    project = _project(tmp_path)
    location_dir = project.path("bible", "locations", "apartment")
    location_dir.mkdir(parents=True, exist_ok=True)
    location_dir.joinpath("location.json").write_text('{"name": "Apartment"}')
    location_dir.joinpath("reference.png").write_bytes(b"apartment")
    location_dir.joinpath("reference.provider-prompt.md").write_text("fitted location")
    location_dir.joinpath("reference.provider-prompt.json").write_text(
        '{"limit": 8000}'
    )

    result = invalidate_artifact_revision(
        project, "bible/locations/apartment/location.json"
    )

    assert "bible/locations/apartment/reference.provider-prompt.md" in result.archived_paths
    assert "bible/locations/apartment/reference.provider-prompt.json" in result.archived_paths


def test_character_reference_regeneration_archives_derived_state_anchors(tmp_path):
    project = _project(tmp_path)
    mara_state = project.path("bible", "characters", "mara", "states", "powered")
    theo_state = project.path("bible", "characters", "theo", "states", "winter")
    mara_state.mkdir(parents=True, exist_ok=True)
    theo_state.mkdir(parents=True, exist_ok=True)
    (mara_state / "reference.png").write_bytes(b"stale-mara-state")
    (mara_state / "reference.prompt.md").write_text("stale Mara state prompt")
    (theo_state / "reference.png").write_bytes(b"current-theo-state")
    (theo_state / "reference.prompt.md").write_text("current Theo state prompt")

    result = invalidate_bible_asset(
        project, "bible/characters/mara/reference.png"
    )

    for rel in (
        "bible/characters/mara/states/powered/reference.png",
        "bible/characters/mara/states/powered/reference.prompt.md",
    ):
        assert not project.path(*rel.split("/")).exists()
        assert _archived(project, result, rel).is_file()
    assert (theo_state / "reference.png").read_bytes() == b"current-theo-state"
    assert (theo_state / "reference.prompt.md").read_text() == "current Theo state prompt"


def test_unknown_reference_target_falls_back_to_all_shots(tmp_path):
    project = _project(tmp_path)

    result = invalidate_reference_change(
        project, target_type="character", target_id="Unknown Person"
    )

    assert result.affected_shot_ids == ["sh-001", "sh-002"]
    assert not project.path("storyboard", "keyframes", "sh-001.png").exists()
    assert not project.path("storyboard", "keyframes", "sh-002.png").exists()
    assert project.current_stage == "storyboard"


def test_character_reference_change_rebuilds_existing_bible_before_shots_exist(tmp_path):
    project = _project(tmp_path)
    project.path("storyboard", "shots.json").unlink()

    result = invalidate_reference_change(
        project, target_type="character", target_id="Mara"
    )

    assert result.affected_shot_ids == []
    assert not project.path("bible", "characters", "mara", "reference.png").exists()
    # turnaround.png is no longer generated/archived; it stays untouched if present
    assert _archived(
        project, result, "bible/characters/mara/reference.png"
    ).is_file()
    assert project.current_stage == "bible"


def test_character_reference_change_invalidates_derived_identity_board(tmp_path):
    project = _project(tmp_path)
    board = project.path("bible", "characters", "mara", "identity_board.json")
    board.write_text("{}")

    result = invalidate_reference_change(
        project, target_type="character", target_id="Mara"
    )

    assert not board.exists()
    assert _archived(
        project, result, "bible/characters/mara/identity_board.json"
    ).is_file()


def test_retarget_reference_invalidates_old_and_new_targets(tmp_path):
    from studio_agent.reference_assets import retarget_reference, save_reference_upload

    project = _project(tmp_path)
    record = save_reference_upload(
        project,
        filename="mara.png",
        data=b"\x89PNG\r\n\x1a\nreference",
        target_type="character",
        target_id="Mara",
    )

    retarget_reference(project, record["id"], "character", "Theo")

    assert not project.path("bible", "characters", "mara", "reference.png").exists()
    assert not project.path("bible", "characters", "theo", "reference.png").exists()
    assert not project.path("storyboard", "keyframes", "sh-001.png").exists()
    assert not project.path("storyboard", "keyframes", "sh-002.png").exists()
    assert project.current_stage == "bible"


def test_character_reference_change_archives_derived_state_anchors(tmp_path):
    project = _project(tmp_path)
    mara_state = project.path("bible", "characters", "mara", "states", "powered")
    theo_state = project.path("bible", "characters", "theo", "states", "winter")
    mara_state.mkdir(parents=True, exist_ok=True)
    theo_state.mkdir(parents=True, exist_ok=True)
    (mara_state / "reference.png").write_bytes(b"stale-mara-state")
    (mara_state / "reference.prompt.md").write_text("stale Mara state prompt")
    (theo_state / "reference.png").write_bytes(b"current-theo-state")
    (theo_state / "reference.prompt.md").write_text("current Theo state prompt")

    result = invalidate_reference_change(
        project, target_type="character", target_id="Mara"
    )

    for rel in (
        "bible/characters/mara/states/powered/reference.png",
        "bible/characters/mara/states/powered/reference.prompt.md",
    ):
        assert not project.path(*rel.split("/")).exists()
        assert _archived(project, result, rel).is_file()
    assert (theo_state / "reference.png").read_bytes() == b"current-theo-state"
    assert (theo_state / "reference.prompt.md").read_text() == "current Theo state prompt"


def test_reference_change_uses_canonical_bible_directory_for_non_latin_name(tmp_path):
    project = _project(tmp_path)
    cdir = project.path("bible", "characters", "character")
    cdir.mkdir(parents=True, exist_ok=True)
    cdir.joinpath("character.json").write_text(json.dumps({"name": "小明"}))
    cdir.joinpath("reference.png").write_bytes(b"xiaoming-reference")
    shots = json.loads(project.path("storyboard", "shots.json").read_text())
    shots["shots"][0]["characters"] = ["小明"]
    shots["shots"][0]["reference_characters"] = ["小明"]
    project.path("storyboard", "shots.json").write_text(
        json.dumps(shots, ensure_ascii=False)
    )

    result = invalidate_reference_change(
        project, target_type="character", target_id="小明"
    )

    assert not cdir.joinpath("reference.png").exists()
    assert _archived(
        project, result, "bible/characters/character/reference.png"
    ).read_bytes() == b"xiaoming-reference"


# ── Clip-mode regression test (I1) ──────────────────────────────────────────

CLIP_PIPELINE = [
    "concept", "bible", "clip", "keyframes", "video_prompts", "video",
    "review", "audio", "assemble",
]


def _clip_project(tmp_path):
    """Build a clip-mode project with storyboard/shots.json present and clip approved."""
    project = Project.create(
        "a dancer spinning under cherry blossoms",
        root=tmp_path,
        stages=CLIP_PIPELINE,
    )
    shots = {
        "shots": [
            {
                "id": "sh-001",
                "keyframe": "sh-001.png",
                "characters": ["Dancer"],
                "reference_characters": ["Dancer"],
                "reference_locations": [],
                "reference_images": [],
            },
        ]
    }
    project.path("storyboard", "shots.json").write_text(json.dumps(shots))
    project.path("storyboard", "keyframes", "sh-001.png").write_bytes(b"keyframe")
    for suffix in ("keyframe.md", "video.md", "audio.md"):
        project.path("storyboard", "prompts", f"sh-001.{suffix}").write_text(
            f"prompt-sh-001-{suffix}"
        )
    project.path("assets", "clips", "sh-001.mp4").write_bytes(b"clip-sh-001")
    project.path("assets", "clips", "sh-001.last_frame.png").write_bytes(b"last")
    project.path("assets", "qc", "sh-001.json").write_text("{}")
    project.path("assets", "clips", "clips.json").write_text("{}")
    project.path("assets", "qc", "summary.json").write_text("{}")
    project.path("edit", "timeline.json").write_text("{}")
    project.path("output", f"{project.project_id}.mp4").write_bytes(b"final")

    # Mark all prior stages approved; clip itself is approved (i.e., everything done)
    for stage in CLIP_PIPELINE:
        project.set_stage_status(stage, "approved")
    project.current_stage = None
    project.status = "done"
    project.save()
    confirm_prompt_batch(project, "keyframes", confirmer="test")
    confirm_prompt_batch(project, "videos", confirmer="test")
    return project


def test_clip_mode_shots_json_revision_invalidates_clip_stage(tmp_path):
    """Editing storyboard/shots.json on a clip-mode project must reset the clip stage to
    pending and set current_stage = 'clip', not return an empty invalidated_stages list."""
    project = _clip_project(tmp_path)

    result = invalidate_artifact_revision(project, "storyboard/shots.json")

    # The invalidated_stages list must be non-empty and start with 'clip'
    assert result.invalidated_stages, (
        "invalidated_stages must not be empty for clip-mode shots.json revision"
    )
    assert result.invalidated_stages[0] == "clip", (
        f"expected invalidated_stages[0]='clip', got {result.invalidated_stages!r}"
    )
    assert project.stage_status("clip") == "pending", (
        f"clip stage must be reset to pending, got {project.stage_status('clip')!r}"
    )


def test_clip_mode_known_video_prompt_revision_returns_to_video_prompt_gate(tmp_path):
    project = _clip_project(tmp_path)

    result = invalidate_artifact_revision(
        project, "storyboard/prompts/sh-001.video.md"
    )

    assert result.affected_shot_ids == ["sh-001"]
    assert result.invalidated_stages[0] == "video_prompts"
    assert project.current_stage == "video_prompts"
    assert project.path("assets", "clips", "sh-001.mp4").is_file()


# ── Motion-grid prompt artifact tests (Task 9) ──────────────────────────────


def test_regenerate_shot_clears_grid_prompt_artifacts(tmp_path):
    """invalidate_shots with include_prompts=True must also clear grid prompt artifacts."""
    from studio_agent.invalidation import invalidate_shots

    project = _project(tmp_path)
    for shot_id in ("sh-001", "sh-002"):
        project.path("storyboard", "prompts", f"{shot_id}.grid.brief.md").write_text(
            f"grid-brief-{shot_id}"
        )
        project.path("storyboard", "prompts", f"{shot_id}.grid.md").write_text(
            f"grid-{shot_id}"
        )

    result = invalidate_shots(
        project,
        ["sh-001"],
        from_stage="storyboard",
        reason="test-grid-clear",
        include_keyframes=False,
        include_prompts=True,
    )

    # grid prompts for sh-001 must be cleared
    assert not project.path("storyboard", "prompts", "sh-001.grid.brief.md").exists()
    assert not project.path("storyboard", "prompts", "sh-001.grid.md").exists()
    assert "storyboard/prompts/sh-001.grid.brief.md" in result.archived_paths
    assert "storyboard/prompts/sh-001.grid.md" in result.archived_paths

    # sh-002 grid prompts must be untouched
    assert project.path("storyboard", "prompts", "sh-002.grid.brief.md").is_file()
    assert project.path("storyboard", "prompts", "sh-002.grid.md").is_file()


def test_grid_prompt_revision_maps_to_keyframe_generation(tmp_path):
    """Editing a *.grid.md file must invalidate the matching keyframe and downstream."""
    project = _project(tmp_path)
    project.path("storyboard", "prompts", "sh-001.grid.md").write_text("directed grid")

    result = invalidate_artifact_revision(
        project, "storyboard/prompts/sh-001.grid.md"
    )

    assert result.affected_shot_ids == ["sh-001"]
    assert "video" in result.invalidated_stages
    assert project.current_stage == "keyframes"
    assert not project.path("storyboard", "keyframes", "sh-001.png").exists()
    assert not project.path("assets", "clips", "sh-001.mp4").exists()


def test_character_reference_png_only_no_removed_sheets(tmp_path):
    """After the combined-sheet refactor, invalidating reference.png must NOT archive
    turnaround.png / expressions.png — those files are no longer generated.
    The only bible file archived should be reference.png itself."""
    project = _project(tmp_path)

    result = invalidate_bible_asset(
        project, "bible/characters/mara/reference.png"
    )

    # reference.png must be archived
    assert "bible/characters/mara/reference.png" in result.archived_paths

    # The removed sheets must NOT appear in the archived paths at all
    assert "bible/characters/mara/turnaround.png" not in result.archived_paths
    assert "bible/characters/mara/expressions.png" not in result.archived_paths

    # Downstream keyframes and clips still cascade (regression guard)
    assert "storyboard/keyframes/sh-001.png" in result.archived_paths
    assert "assets/clips/sh-001.mp4" in result.archived_paths
    assert result.affected_shot_ids == ["sh-001"]


def test_location_reference_png_only_no_environment_board(tmp_path):
    """After the combined-sheet refactor, invalidating a location reference.png must NOT
    archive environment_board.png — that file is no longer generated.

    Importantly, environment_board.png IS created on disk so the assertion is non-vacuous:
    if production code still listed it, the on-disk file would be archived and the test
    would fail.
    """
    project = _project(tmp_path)
    loc_base = project.path("bible", "locations", "apartment")
    loc_base.mkdir(parents=True, exist_ok=True)
    loc_base.joinpath("reference.png").write_bytes(b"loc-reference")
    # Create environment_board.png on disk so the assertion below is non-vacuous.
    env_board = loc_base.joinpath("environment_board.png")
    env_board.write_bytes(b"env-board-data")

    result = invalidate_bible_asset(
        project, "bible/locations/apartment/reference.png"
    )

    # reference.png must be archived (the file that triggered the invalidation)
    assert "bible/locations/apartment/reference.png" in result.archived_paths

    # environment_board.png must NOT be archived — it is no longer part of the cascade
    assert "bible/locations/apartment/environment_board.png" not in result.archived_paths
    # Confirm the file itself still lives on disk (was not moved to the archive)
    assert env_board.is_file(), "environment_board.png must remain on disk, not be archived"

    # Downstream cascade: Apartment is referenced by sh-001 only
    assert result.affected_shot_ids == ["sh-001"]
    assert "storyboard/keyframes/sh-001.png" in result.archived_paths
    assert "assets/clips/sh-001.mp4" in result.archived_paths
    # sh-002 (Street) must be untouched
    assert project.path("assets", "clips", "sh-002.mp4").is_file()
    assert project.stage_status("bible") == "pending"
    assert project.current_stage == "bible"


def test_invalidation_module_has_no_removed_sheet_filenames(tmp_path):
    """Source-level guard: the invalidation module must not reference the removed filenames."""
    from pathlib import Path
    import studio_agent.invalidation as inv

    text = Path(inv.__file__).read_text()
    assert "turnaround.png" not in text, "turnaround.png still referenced in invalidation.py"
    assert "expressions.png" not in text, "expressions.png still referenced in invalidation.py"
    assert "environment_board.png" not in text, "environment_board.png still referenced in invalidation.py"


def test_refresh_shot_packet_invalidates_only_its_shot(tmp_path):
    project = _project(tmp_path)
    packet = project.path("knowledge", "packets", "shot-sh-001.json")
    packet.parent.mkdir(parents=True, exist_ok=True)
    packet.write_text(json.dumps({"purpose": "shot", "target": "sh-001"}))
    project.path("storyboard", "prompts", "sh-001.knowledge.json").write_text("{}")
    project.path("storyboard", "prompts", "sh-001.video.md").write_text("x")
    untouched = project.path("assets", "clips", "sh-002.mp4").read_bytes()

    result = invalidation.refresh_knowledge_target(
        project,
        purpose="shot",
        target="sh-001",
    )

    assert result.affected_shot_ids == ["sh-001"]
    assert not packet.exists()
    assert not project.path("storyboard", "prompts", "sh-001.knowledge.json").exists()
    assert not project.path("storyboard", "prompts", "sh-001.video.md").exists()
    assert not project.path("assets", "clips", "sh-001.mp4").exists()
    assert project.path("assets", "clips", "sh-002.mp4").read_bytes() == untouched


def test_invalidate_bible_asset_regrounds_board_when_reference_attached(tmp_path):
    # Closing the loop: regenerating a character image whose identity came from an uploaded
    # reference must also archive the identity board so it re-derives from the image — so the
    # new sheet and the board agree instead of fighting a stale, image-blind board.
    from studio_agent.reference_assets import save_reference_intake_uploads

    project = _project(tmp_path)
    char_dir = project.path("bible", "characters", "mara")
    (char_dir / "identity_board.json").write_text("{}")
    save_reference_intake_uploads(
        project,
        [{"filename": "mara.png", "data": b"\x89PNG\r\n\x1a\nmara", "content_type": "image/png"}],
        target_type="character", target_id="Mara",
    )

    invalidate_bible_asset(project, "bible/characters/mara/reference.png")

    assert not (char_dir / "identity_board.json").is_file()


def test_invalidate_bible_asset_keeps_board_without_reference(tmp_path):
    # No uploaded reference: image regen leaves the board alone (existing behavior).
    project = _project(tmp_path)
    char_dir = project.path("bible", "characters", "mara")
    (char_dir / "identity_board.json").write_text("{}")

    invalidate_bible_asset(project, "bible/characters/mara/reference.png")

    assert (char_dir / "identity_board.json").is_file()


def test_invalidate_bible_text_keeps_image_and_downstream(tmp_path):
    project = _project(tmp_path)
    # Set every stage to complete so we can confirm downstream is NOT reset
    for stage in PIPELINE:
        project.set_stage_status(stage, "complete")
    project.save()
    # Write the derived text file that should be archived
    char_dir = project.path("bible", "characters", "mara")
    (char_dir / "identity_board.json").write_text("{}")
    # Verify reference.png already exists from _project builder
    assert (char_dir / "reference.png").is_file()
    # Verify downstream keyframe exists
    keyframe = project.path("storyboard", "keyframes", "sh-001.png")
    assert keyframe.is_file()

    result = invalidate_bible_text(project, "character", "mara")

    assert not (char_dir / "identity_board.json").is_file()  # derived text archived
    assert (char_dir / "reference.png").is_file()            # image untouched
    assert keyframe.is_file()                                 # downstream untouched
    reloaded = Project.load(project.dir)
    assert reloaded.stage_status("bible") == "pending"
    assert reloaded.stage_status("storyboard") == "complete"
