"""Scoped asset regeneration helpers for the review workbench."""

import json

import pytest

from studio_agent.asset_regeneration import (
    delete_clip_shot,
    pending_regeneration,
    regenerate_bible_asset,
    regenerate_bible_text,
    regenerate_keyframe,
    regenerate_shot_video,
    restore_archived_shot_family,
    revise_keyframe_from_feedback,
    revise_video_from_feedback,
    set_clip_duration,
)
from studio_agent.orchestrator.project import CostCapExceeded, Project
from studio_agent.prompt_approvals import confirm_prompt_batch
from studio_agent.providers.fake import (
    FakeImageGen,
    FakeLLM,
    FakeReferenceAnalyzer,
    FakeVideoGen,
)
from studio_agent.stages.base import Providers
from studio_agent.stages.keyframes import KeyframesStage
from studio_agent.stages.video import VideoStage


PIPELINE = [
    "plot", "script", "bible", "storyboard", "keyframes", "video_prompts",
    "video", "review", "audio", "assemble",
]


def _project(tmp_path):
    p = Project.create("asset regen", root=tmp_path, stages=PIPELINE)
    p.path("storyboard", "shots.json").write_text(json.dumps({"shots": [
        {
            "id": "sh-001",
            "keyframe": "sh-001.png",
            "keyframe_prompt": "old prompt one",
            "characters": ["Mara"],
            "reference_characters": ["Mara"],
        },
        {
            "id": "sh-002",
            "keyframe": "sh-002.png",
            "keyframe_prompt": "old prompt two",
            "characters": ["Theo"],
            "reference_characters": ["Theo"],
        },
    ]}))
    for stage in PIPELINE:
        p.set_stage_status(stage, "approved")
    p.current_stage = None
    p.status = "done"

    cdir = p.path("bible", "characters", "mara")
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "reference.png").write_bytes(b"reference.png")
    state_dir = cdir / "states" / "powered"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "reference.png").write_bytes(b"powered")

    for shot_id in ("sh-001", "sh-002"):
        p.path("storyboard", "prompts", f"{shot_id}.keyframe.md").write_text(f"old prompt {shot_id}")
        p.path("storyboard", "prompts", f"{shot_id}.video.md").write_text(f"old video prompt {shot_id}")
        p.path("storyboard", "keyframes", f"{shot_id}.png").write_bytes(f"kf {shot_id}".encode())
        p.path("assets", "clips", f"{shot_id}.mp4").write_bytes(f"clip {shot_id}".encode())
        p.path("assets", "clips", f"{shot_id}.last_frame.png").write_bytes(b"last")
        p.path("assets", "qc", f"{shot_id}.json").write_text(json.dumps({"shot": shot_id}))
        p.path("assets", "audio", f"{shot_id}.dialogue.wav").write_bytes(b"audio")
    p.path("assets", "qc", "summary.json").write_text(json.dumps({"ok": True}))
    p.path("edit", "timeline.json").write_text(json.dumps({"ok": True}))
    p.path("output", f"{p.project_id}.mp4").write_bytes(b"final")
    p.save()
    return p


def test_regenerate_character_asset_rotates_seed_so_rerender_differs(tmp_path):
    # Bug: the sheet seed is a constant derived from the name, so regenerating produced the
    # same deterministic image and "didn't take". Explicit regeneration must rotate the seed
    # (and bump a counter) so the resume renders a genuinely new sheet that replaces the old.
    p = _project(tmp_path)
    cdir = p.path("bible", "characters", "mara")
    cdir.joinpath("character.json").write_text(
        json.dumps({"name": "Mara", "seed": 1234}), encoding="utf-8"
    )

    regenerate_bible_asset(p, "bible/characters/mara/reference.png")

    data = json.loads(cdir.joinpath("character.json").read_text())
    assert data["seed"] != 1234, "seed must change so the regenerated sheet differs"
    assert data.get("regen_count") == 1

    # The resume would re-render reference.png from the new seed; simulate that before regen #2.
    cdir.joinpath("reference.png").write_bytes(b"reference.png v2")
    regenerate_bible_asset(p, "bible/characters/mara/reference.png")
    data2 = json.loads(cdir.joinpath("character.json").read_text())
    assert data2["seed"] not in (1234, data["seed"]), "each regen rotates to a fresh seed"
    assert data2.get("regen_count") == 2


def test_regenerate_bible_asset_archives_derivatives_and_affected_shot(tmp_path):
    p = _project(tmp_path)

    result = regenerate_bible_asset(p, "bible/characters/mara/reference.png")

    assert result.removed >= 1
    assert result.archive_dir
    assert p.path("bible", "characters", "mara", "reference.png").exists() is False
    assert p.path("storyboard", "keyframes", "sh-001.png").exists() is False
    assert p.path("assets", "clips", "sh-001.mp4").exists() is False
    assert p.path("storyboard", "keyframes", "sh-002.png").is_file()
    assert p.path("assets", "clips", "sh-002.mp4").is_file()
    archived_reference = p.dir / result.archive_dir / "bible/characters/mara/reference.png"
    assert archived_reference.read_bytes() == b"reference.png"
    assert result.invalidated == [
        "bible", "storyboard", "keyframes", "video_prompts", "video", "review",
        "audio", "assemble",
    ]
    assert p.current_stage == "bible"
    assert p.stage_status("plot") == "approved"
    assert p.stage_status("bible") == "pending"
    assert p.stage_status("assemble") == "pending"


def test_regenerate_derived_state_reference_keeps_baseline_identity(tmp_path):
    p = _project(tmp_path)

    result = regenerate_bible_asset(
        p, "bible/characters/mara/states/powered/reference.png"
    )

    assert result.removed >= 1
    assert p.path("bible", "characters", "mara", "reference.png").is_file()
    assert not p.path(
        "bible", "characters", "mara", "states", "powered", "reference.png"
    ).exists()
    assert not p.path("storyboard", "keyframes", "sh-001.png").exists()
    assert p.current_stage == "bible"


def test_regenerate_keyframe_queues_candidate_and_keeps_accepted_outputs(tmp_path):
    p = _project(tmp_path)

    result = regenerate_keyframe(p, "sh-001")

    assert result.removed == 0
    assert not result.archive_dir
    assert p.path("storyboard", "keyframes", "sh-001.png").is_file()
    assert p.path("assets", "clips", "sh-001.mp4").is_file()
    assert p.path("assets", "clips", "sh-001.last_frame.png").is_file()
    assert p.path("assets", "qc", "summary.json").is_file()
    assert p.path("edit", "timeline.json").is_file()
    assert p.path("output", f"{p.project_id}.mp4").is_file()
    assert pending_regeneration(p, "sh-001", "keyframe")

    assert p.path("storyboard", "keyframes", "sh-002.png").is_file()
    assert p.path("assets", "clips", "sh-002.mp4").is_file()
    assert p.current_stage == "keyframes"
    assert result.invalidated == [
        "keyframes", "video_prompts", "video", "review", "audio", "assemble",
    ]


def test_revise_keyframe_from_feedback_updates_prompt_then_queues_candidate(tmp_path):
    p = _project(tmp_path)

    result = revise_keyframe_from_feedback(
        p,
        "sh-001",
        "make the face less photoreal and keep the red coat",
        Providers(llm=FakeLLM()),
    )

    prompt = p.path("storyboard", "prompts", "sh-001.keyframe.md").read_text()
    shots = json.loads(p.path("storyboard", "shots.json").read_text())["shots"]
    shot_one = next(shot for shot in shots if shot["id"] == "sh-001")
    shot_two = next(shot for shot in shots if shot["id"] == "sh-002")

    assert result.applied is True
    assert result.prompt_path == "storyboard/prompts/sh-001.keyframe.md"
    assert result.removed == 0
    assert "Revision note:" in prompt
    assert "less photoreal" in prompt
    assert shot_one["keyframe_prompt"] == prompt.strip()
    assert shot_two["keyframe_prompt"] == "old prompt two"
    assert p.path("storyboard", "keyframes", "sh-001.png").is_file()
    assert p.path("storyboard", "keyframes", "sh-002.png").is_file()
    assert p.current_stage == "keyframes"
    assert result.invalidated == []
    assert pending_regeneration(p, "sh-001", "keyframe")
    assert p.cost_log[-1]["stage"] == "adjustment"
    assert list(p.path("adjustments").glob("*.json"))


def test_revise_video_from_feedback_updates_prompt_without_rerendering(tmp_path):
    # Cheap LLM edit of the motion prompt only; the paid clip stays until the user clicks
    # Regenerate (invariant #7). The clip family must be untouched by the revise step.
    p = _project(tmp_path)
    p.path("storyboard", "prompts", "sh-001.video.md").write_text("old video prompt one")

    result = revise_video_from_feedback(
        p,
        "sh-001",
        "slow the camera push and keep the dialogue exact",
        Providers(llm=FakeLLM()),
    )

    prompt = p.path("storyboard", "prompts", "sh-001.video.md").read_text()
    assert result.applied is True
    assert result.prompt_path == "storyboard/prompts/sh-001.video.md"
    assert "slow the camera push" in prompt
    assert result.removed == 0
    assert result.invalidated == []
    assert p.path("assets", "clips", "sh-001.mp4").is_file()      # no paid re-render
    assert p.path("storyboard", "keyframes", "sh-001.png").is_file()
    assert p.cost_log[-1]["stage"] == "adjustment"


def test_revise_video_from_feedback_requires_feedback(tmp_path):
    p = _project(tmp_path)
    p.path("storyboard", "prompts", "sh-001.video.md").write_text("old")
    with pytest.raises(ValueError):
        revise_video_from_feedback(p, "sh-001", "   ", Providers(llm=FakeLLM()))


def test_revise_video_from_feedback_requires_existing_video_prompt(tmp_path):
    p = _project(tmp_path)
    p.path("storyboard", "prompts", "sh-001.video.md").unlink()
    with pytest.raises(FileNotFoundError):
        revise_video_from_feedback(p, "sh-001", "slower push", Providers(llm=FakeLLM()))


def test_video_feedback_checks_named_upload_before_llm_revision(tmp_path):
    from studio_agent.reference_assets import save_reference_upload

    p = _project(tmp_path)
    p.path("storyboard", "prompts", "sh-001.video.md").write_text("old")
    record = save_reference_upload(
        p,
        filename="mara.png",
        data=b"\x89PNG\r\n\x1a\nmissing",
        target_type="character",
        target_id="Mara",
    )
    p.path(*record["path"].split("/")).unlink()

    class _RecordingLLM(FakeLLM):
        def __init__(self):
            self.calls = []

        def complete(self, prompt, *, system=None):
            self.calls.append(prompt)
            return super().complete(prompt, system=system)

    llm = _RecordingLLM()
    with pytest.raises(FileNotFoundError, match=record["path"]):
        revise_video_from_feedback(
            p,
            "sh-001",
            "keep @image1 exactly",
            Providers(llm=llm),
        )

    assert llm.calls == []


def test_revise_then_regenerate_shot_video_renders_from_revised_prompt(tmp_path):
    p = _project(tmp_path)
    p.path("storyboard", "prompts", "sh-001.video.md").write_text("old video prompt one")

    revise_video_from_feedback(
        p, "sh-001", "slow the camera push", Providers(llm=FakeLLM())
    )
    revised = p.path("storyboard", "prompts", "sh-001.video.md").read_text()

    regenerate_shot_video(p, "sh-001")

    # The accepted clip stays live while a candidate is queued from the revised prompt.
    assert p.path("assets", "clips", "sh-001.mp4").is_file()
    assert pending_regeneration(p, "sh-001", "video")
    assert p.path("storyboard", "prompts", "sh-001.video.md").read_text() == revised
    assert "slow the camera push" in revised


def test_regenerate_shot_video_keeps_accepted_family_until_candidate_succeeds(tmp_path):
    p = _project(tmp_path)

    result = regenerate_shot_video(p, "sh-001")

    assert result.removed == 0
    assert not result.archive_dir
    assert p.path("storyboard", "keyframes", "sh-001.png").is_file()
    assert p.path("assets", "clips", "sh-001.mp4").is_file()
    assert p.path("assets", "clips", "sh-002.mp4").is_file()
    assert p.current_stage == "video"
    assert result.invalidated == ["video", "review", "audio", "assemble"]
    assert pending_regeneration(p, "sh-001", "video")


def _approve_regeneration_prompts(project):
    for shot in json.loads(project.path("storyboard", "shots.json").read_text())["shots"]:
        shot_id = shot["id"]
        project.path("storyboard", "prompts", f"{shot_id}.keyframe.md").write_text(
            f"Static medium shot of {shot_id} standing beside a window."
        )
        project.path("storyboard", "prompts", f"{shot_id}.video.md").write_text(
            f"{shot_id} turns toward the window while the camera slowly pushes in."
        )
    confirm_prompt_batch(project, "keyframes", confirmer="test")
    confirm_prompt_batch(project, "videos", confirmer="test")


def test_regeneration_over_cap_keeps_accepted_clip_and_writes_no_request(tmp_path):
    project = _project(tmp_path)
    _approve_regeneration_prompts(project)
    project.cost_cap = 1.0
    project.add_cost(stage="video", provider="fake", cost_usd=1.0, seconds=1)
    old = project.path("assets", "clips", "sh-001.mp4").read_bytes()

    with pytest.raises(CostCapExceeded):
        regenerate_shot_video(project, "sh-001")

    assert project.path("assets", "clips", "sh-001.mp4").read_bytes() == old
    assert pending_regeneration(project, "sh-001", "video") is None


def test_video_provider_failure_keeps_accepted_clip(tmp_path):
    project = _project(tmp_path)
    _approve_regeneration_prompts(project)
    old = project.path("assets", "clips", "sh-001.mp4").read_bytes()
    regenerate_shot_video(project, "sh-001")

    class FailingVideo(FakeVideoGen):
        def generate(self, prompt, *, out_path, **kwargs):
            raise RuntimeError("provider failed")

    with pytest.raises(RuntimeError, match="provider failed"):
        VideoStage().run(project, Providers(video=FailingVideo()))

    assert project.path("assets", "clips", "sh-001.mp4").read_bytes() == old
    assert "provider failed" in pending_regeneration(project, "sh-001", "video")["error"]


def test_successful_video_candidate_archives_old_and_promotes_new(tmp_path):
    project = _project(tmp_path)
    _approve_regeneration_prompts(project)
    old = project.path("assets", "clips", "sh-001.mp4").read_bytes()
    regenerate_shot_video(project, "sh-001")

    VideoStage(frame_extractor=lambda *_args: False).run(
        project, Providers(video=FakeVideoGen())
    )

    assert project.path("assets", "clips", "sh-001.mp4").read_bytes() != old
    assert list(project.path("history").rglob("sh-001.mp4"))
    assert pending_regeneration(project, "sh-001", "video") is None


def test_keyframe_provider_failure_keeps_accepted_image(tmp_path):
    project = _project(tmp_path)
    _approve_regeneration_prompts(project)
    old = project.path("storyboard", "keyframes", "sh-001.png").read_bytes()
    regenerate_keyframe(project, "sh-001")

    class FailingImage(FakeImageGen):
        def generate(self, prompt, *, out_path, **kwargs):
            raise RuntimeError("image provider failed")

    with pytest.raises(RuntimeError, match="image provider failed"):
        KeyframesStage().run(project, Providers(image=FailingImage()))

    assert project.path("storyboard", "keyframes", "sh-001.png").read_bytes() == old


def test_successful_keyframe_candidate_archives_old_and_promotes_new(tmp_path):
    project = _project(tmp_path)
    _approve_regeneration_prompts(project)
    old = project.path("storyboard", "keyframes", "sh-001.png").read_bytes()
    regenerate_keyframe(project, "sh-001")

    KeyframesStage().run(project, Providers(image=FakeImageGen()))

    assert project.path("storyboard", "keyframes", "sh-001.png").read_bytes() != old
    assert list(project.path("history").rglob("sh-001.png"))
    assert pending_regeneration(project, "sh-001", "keyframe") is None


def test_regenerate_native_shot_video_keeps_accepted_audio_until_promotion(tmp_path):
    p = _project(tmp_path)
    p.model_config["audio_mode"] = "native_video"
    for shot_id in ("sh-001", "sh-002"):
        p.path("assets", "clips", f"{shot_id}.native.source.wav").write_bytes(
            f"source {shot_id}".encode()
        )
        p.path("assets", "audio", f"{shot_id}.native.wav").write_bytes(
            f"native {shot_id}".encode()
        )
    p.path("assets", "audio", "audio.json").write_text("{}")

    result = regenerate_shot_video(p, "sh-001")

    assert p.path("assets", "clips", "sh-001.native.source.wav").is_file()
    assert p.path("assets", "audio", "sh-001.native.wav").is_file()
    assert p.path("assets", "audio", "audio.json").is_file()
    assert p.path("assets", "clips", "sh-002.native.source.wav").is_file()
    assert p.path("assets", "audio", "sh-002.native.wav").is_file()
    assert result.archive_dir == ""
    assert pending_regeneration(p, "sh-001", "video")


def test_regenerate_missing_asset_does_not_mutate_project(tmp_path):
    p = _project(tmp_path)

    with pytest.raises(FileNotFoundError):
        regenerate_bible_asset(p, "bible/characters/missing/reference.png")

    assert p.current_stage is None
    assert p.stage_status("bible") == "approved"


def test_restore_archived_shot_family_restores_only_missing_files(tmp_path):
    p = _project(tmp_path)
    archive = p.path("history", "offline-recovery", "assets", "clips")
    archive.mkdir(parents=True)
    archive.joinpath("sh-001.mp4").write_bytes(b"archived clip")
    archive.joinpath("sh-001.last_frame.png").write_bytes(b"archived last")
    archive.joinpath("clips.json").write_text('{"clips": [{"id": "sh-001"}]}')
    p.path("history", "offline-recovery", "manifest.json").write_text(json.dumps({
        "affected_shot_ids": ["sh-001"],
        "archived_paths": [
            "assets/clips/sh-001.mp4",
            "assets/clips/sh-001.last_frame.png",
            "assets/clips/clips.json",
        ],
    }))
    p.path("assets", "clips", "sh-001.mp4").unlink()
    p.path("assets", "clips", "clips.json").unlink(missing_ok=True)
    accepted_last = p.path("assets", "clips", "sh-001.last_frame.png").read_bytes()

    restored = restore_archived_shot_family(
        p, "history/offline-recovery", "sh-001"
    )

    assert restored == ["assets/clips/clips.json", "assets/clips/sh-001.mp4"]
    assert p.path("assets", "clips", "sh-001.mp4").read_bytes() == b"archived clip"
    assert p.path("assets", "clips", "clips.json").is_file()
    assert p.path("assets", "clips", "sh-001.last_frame.png").read_bytes() == accepted_last


def test_restore_archived_shot_family_rejects_outside_history(tmp_path):
    p = _project(tmp_path)
    with pytest.raises(ValueError, match="inside project history"):
        restore_archived_shot_family(p, "assets/clips", "sh-001")


def test_validate_bible_asset_rejects_removed_character_sheets(tmp_path):
    """turnaround.png and expressions.png no longer exist; the gate must reject them."""
    from studio_agent.asset_regeneration import _validate_bible_asset_rel

    with pytest.raises(ValueError, match="unsupported bible generated asset"):
        _validate_bible_asset_rel("bible/characters/mara/turnaround.png")

    with pytest.raises(ValueError, match="unsupported bible generated asset"):
        _validate_bible_asset_rel("bible/characters/mara/expressions.png")


def test_validate_bible_asset_rejects_removed_location_sheet(tmp_path):
    """environment_board.png no longer exists; the gate must reject it."""
    from studio_agent.asset_regeneration import _validate_bible_asset_rel

    with pytest.raises(ValueError, match="unsupported bible generated asset"):
        _validate_bible_asset_rel("bible/locations/lighthouse/environment_board.png")


def test_validate_bible_asset_accepts_combined_reference_png(tmp_path):
    """The combined reference.png sheet must still be accepted for both entity types."""
    from studio_agent.asset_regeneration import _validate_bible_asset_rel

    assert _validate_bible_asset_rel("bible/characters/mara/reference.png") == \
        "bible/characters/mara/reference.png"
    assert _validate_bible_asset_rel("bible/locations/lighthouse/reference.png") == \
        "bible/locations/lighthouse/reference.png"


# ---------------------------------------------------------------------------
# regenerate_bible_text tests
# ---------------------------------------------------------------------------

def _bible_project_with_fake_llm(tmp_path):
    """Build a project with a character having text + image artifacts, plus a fake LLM."""
    p = _project(tmp_path)
    char_dir = p.path("bible", "characters", "mara")
    # char_dir already exists (created by _project); write text + board artifacts
    p.path("bible", "characters", "mara", "character.json").write_text(
        json.dumps({"name": "Mara", "description": "a woman"})
    )
    p.path("bible", "characters", "mara", "identity_board.json").write_text("{}")
    return (p, Providers(llm=FakeLLM()))


def test_regenerate_bible_text_revises_text_and_keeps_image(tmp_path):
    project, providers = _bible_project_with_fake_llm(tmp_path)
    ref = project.path("bible", "characters", "mara", "reference.png")
    board = project.path("bible", "characters", "mara", "identity_board.json")
    assert ref.is_file() and board.is_file()

    result = regenerate_bible_text(project, "character", "mara", "make her older", providers)

    assert result.applied is True
    assert result.needs_resume is True
    assert result.rel == "bible/characters/mara/character.json"
    assert ref.is_file()            # image untouched
    assert not board.is_file()      # derived board archived for rebuild
    assert Project.load(project.dir).stage_status("bible") == "pending"


def test_regenerate_bible_text_rejects_empty_instruction(tmp_path):
    project, providers = _bible_project_with_fake_llm(tmp_path)
    with pytest.raises(ValueError):
        regenerate_bible_text(project, "character", "mara", "   ", providers)


def test_regenerate_bible_text_grounds_in_subject_reference(tmp_path):
    import json
    from studio_agent.asset_regeneration import regenerate_bible_text
    from studio_agent.reference_assets import save_reference_upload

    p = _project(tmp_path)
    cdir = p.path("bible", "characters", "mara")
    (cdir / "character.json").write_text(json.dumps({"name": "Mara", "wardrobe": ""}))
    save_reference_upload(
        p, filename="mara.png", data=b"\x89PNG\r\n\x1a\nm",
        target_type="character", target_id="Mara",
    )

    regenerate_bible_text(
        p, "character", "mara", "keep the same outfit",
        Providers(llm=FakeLLM(), reference_analyzer=FakeReferenceAnalyzer()),
    )

    revised = json.loads((cdir / "character.json").read_text())
    # grounded_in contains the file stem of the subject upload (includes uuid prefix from
    # _write_upload_file: "{uuid}-mara"); check that exactly one grounding occurred and
    # the original filename stem "mara" is present in the stored stem.
    grounded = revised.get("grounded_in", [])
    assert len(grounded) == 1 and "mara" in grounded[0]  # the subject upload was shown to the vision model


# ---------------------------------------------------------------------------
# delete_clip_shot tests
# ---------------------------------------------------------------------------


def _clip_project(tmp_path):
    """Clip-mode project paused at the clip gate with two planned clips."""
    # Matches cli.CLIP_PIPELINE.
    stages = [
        "style", "concept", "bible", "clip", "keyframes", "video_prompts",
        "video", "audio", "assemble",
    ]
    p = Project.create("clip gate", root=tmp_path, stages=stages)
    p.path("storyboard", "shots.json").write_text(json.dumps({"shots": [
        {"id": "s01", "keyframe": "s01.png", "duration_s": 12, "deps": [],
         "continuity_reference": "", "keyframe_reference_images": []},
        {"id": "s02", "keyframe": "s02.png", "duration_s": 10, "deps": ["s01"],
         "continuity_reference": "storyboard/keyframes/s01.png",
         "keyframe_reference_images": ["storyboard/keyframes/s01.png"]},
    ]}))
    p.path("storyboard", "visual_sequence.json").write_text(json.dumps({
        "version": 1,
        "target_duration_s": 22,
        "clip_durations": [12, 10],
        "beats": [{"id": "b1"}, {"id": "b2"}],
        "segments": [
            {"id": "segment-01", "duration_s": 12, "visual_beats": [{"id": "b1"}]},
            {"id": "segment-02", "duration_s": 10, "visual_beats": [{"id": "b2"}]},
        ],
    }))
    for sid in ("s01", "s02"):
        p.path("storyboard", "keyframes", f"{sid}.png").write_bytes(f"kf {sid}".encode())
        p.path("storyboard", "prompts", f"{sid}.keyframe.md").write_text(f"prompt {sid}")
    for stage in ("style", "concept", "bible"):
        p.set_stage_status(stage, "approved")
    p.set_stage_status("clip", "complete")
    p.current_stage = "clip"
    p.status = "in_progress"
    p.save()
    return p


def test_delete_clip_removes_shot_and_archives_keyframe(tmp_path):
    p = _clip_project(tmp_path)
    result = delete_clip_shot(p, "s02")
    shots = json.loads(p.path("storyboard", "shots.json").read_text())["shots"]
    assert [s["id"] for s in shots] == ["s01"]
    assert not p.path("storyboard", "keyframes", "s02.png").is_file()
    assert list(p.dir.joinpath("history").rglob("s02.png")), "deleted keyframe archived"
    assert result.removed >= 1
    # Clip stage stays at its gate; survivor keyframe untouched.
    reloaded = Project.load(p.dir)
    assert reloaded.stage_status("clip") == "complete"
    assert reloaded.current_stage == "clip"
    assert p.path("storyboard", "keyframes", "s01.png").is_file()
    sequence = json.loads(p.path("storyboard", "visual_sequence.json").read_text())
    assert [segment["id"] for segment in sequence["segments"]] == ["segment-01"]
    assert sequence["clip_durations"] == [12]
    assert sequence["target_duration_s"] == 12


def test_delete_invalidates_immediate_successor_approval_only(tmp_path):
    p = _clip_project(tmp_path)
    shots_path = p.path("storyboard", "shots.json")
    shots = json.loads(shots_path.read_text())["shots"]
    shots.append({
        "id": "s03", "keyframe": "s03.png", "duration_s": 8, "deps": ["s02"],
        "continuity_reference": "storyboard/keyframes/s02.png",
        "keyframe_reference_images": ["storyboard/keyframes/s02.png"],
    })
    shots_path.write_text(json.dumps({"shots": shots}))
    p.path("storyboard", "keyframes", "s03.png").write_bytes(b"kf s03")
    p.path("storyboard", "prompts", "s03.keyframe.md").write_text(
        "Static medium shot of subject three beside a window."
    )
    for sid in ("s01", "s02"):
        p.path("storyboard", "prompts", f"{sid}.keyframe.md").write_text(
            f"Static medium shot of {sid} beside a window."
        )
    sequence = json.loads(p.path("storyboard", "visual_sequence.json").read_text())
    sequence["segments"].append({
        "id": "segment-03", "duration_s": 8, "visual_beats": [{"id": "b3"}]
    })
    sequence["clip_durations"].append(8)
    sequence["beats"].append({"id": "b3"})
    sequence["target_duration_s"] = 30
    p.path("storyboard", "visual_sequence.json").write_text(json.dumps(sequence))
    confirm_prompt_batch(p, "keyframes", confirmer="test")

    delete_clip_shot(p, "s01")

    manifest = json.loads(
        p.path("storyboard", "prompt_approvals", "keyframes.json").read_text()
    )
    assert "s02" not in manifest["prompts"]
    assert "s03" in manifest["prompts"]


def test_delete_first_clip_clears_dangling_continuity(tmp_path):
    p = _clip_project(tmp_path)
    delete_clip_shot(p, "s01")
    shots = json.loads(p.path("storyboard", "shots.json").read_text())["shots"]
    assert [s["id"] for s in shots] == ["s02"]
    survivor = shots[0]
    assert survivor["deps"] == []
    assert survivor.get("continuity_reference", "") == ""
    assert "storyboard/keyframes/s01.png" not in (survivor.get("keyframe_reference_images") or [])


def test_delete_last_clip_is_refused(tmp_path):
    p = _clip_project(tmp_path)
    delete_clip_shot(p, "s02")
    with pytest.raises(ValueError):
        delete_clip_shot(p, "s01")


def test_delete_unknown_clip_raises(tmp_path):
    p = _clip_project(tmp_path)
    with pytest.raises(FileNotFoundError):
        delete_clip_shot(p, "s99")


def test_set_clip_duration_clamps_high(tmp_path):
    p = _clip_project(tmp_path)
    set_clip_duration(p, "s01", 99)
    shots = json.loads(p.path("storyboard", "shots.json").read_text())["shots"]
    assert next(s for s in shots if s["id"] == "s01")["duration_s"] == 15


def test_set_clip_duration_clamps_low(tmp_path):
    # Bounds come from the project's video-provider capabilities (default min is 1).
    p = _clip_project(tmp_path)
    set_clip_duration(p, "s01", 0)
    shots = json.loads(p.path("storyboard", "shots.json").read_text())["shots"]
    assert next(s for s in shots if s["id"] == "s01")["duration_s"] == 1


def test_set_clip_duration_blank_means_provider_default(tmp_path):
    p = _clip_project(tmp_path)
    set_clip_duration(p, "s01", "")
    shots = json.loads(p.path("storyboard", "shots.json").read_text())["shots"]
    assert next(s for s in shots if s["id"] == "s01")["duration_s"] is None


def test_set_clip_duration_in_range(tmp_path):
    p = _clip_project(tmp_path)
    set_clip_duration(p, "s02", 7)
    shots = json.loads(p.path("storyboard", "shots.json").read_text())["shots"]
    assert next(s for s in shots if s["id"] == "s02")["duration_s"] == 7


def test_set_clip_duration_unknown_raises(tmp_path):
    p = _clip_project(tmp_path)
    with pytest.raises(FileNotFoundError):
        set_clip_duration(p, "s99", 8)


def test_keyframe_feedback_naming_an_alias_attaches_that_reference_image(tmp_path):
    # Defect D: "make it look like @image1" must actually condition the re-render on the
    # uploaded image. Rewriting only the prompt text leaves the image model with no image to
    # match, so the referenced upload's path must be added to the shot's reference_images.
    from studio_agent.reference_assets import save_reference_upload

    p = _project(tmp_path)
    record = save_reference_upload(
        p,
        filename="hero.png",
        data=b"\x89PNG\r\n\x1a\nfake",
        target_type="character",
        target_id="Mara",
    )
    assert record["alias"] == "@image1"

    revise_keyframe_from_feedback(
        p,
        "sh-001",
        "make it look like @image1",
        Providers(llm=FakeLLM(), reference_analyzer=FakeReferenceAnalyzer()),
    )

    shots = json.loads(p.path("storyboard", "shots.json").read_text())["shots"]
    shot_one = next(s for s in shots if s["id"] == "sh-001")
    assert record["path"] in (shot_one.get("reference_images") or [])
    assert shot_one["named_reference_images"] == [record["path"]]
    shot_two = next(s for s in shots if s["id"] == "sh-002")
    assert record["path"] not in (shot_two.get("reference_images") or [])


def test_keyframe_feedback_grounds_in_named_alias_and_shot_characters(tmp_path):
    from studio_agent.reference_assets import save_reference_upload

    p = _project(tmp_path)
    record = save_reference_upload(
        p, filename="hero.png", data=b"\x89PNG\r\n\x1a\nh",
        target_type="character", target_id="Mara",
    )

    class _RecordingAnalyzer(FakeReferenceAnalyzer):
        def __init__(self):
            self.seen = []

        def revise(self, image_paths, *, prompt, language="en"):
            self.seen.append(list(image_paths))
            return super().revise(image_paths, prompt=prompt, language=language)

    analyzer = _RecordingAnalyzer()
    revise_keyframe_from_feedback(
        p, "sh-001", "make it look like @image1",
        Providers(llm=FakeLLM(), reference_analyzer=analyzer),
    )

    assert analyzer.seen, "the vision revise path was not used"
    assert any(record["path"].split("/")[-1] in path for path in analyzer.seen[0])


def test_keyframe_feedback_vision_sees_character_and_location_uploads(tmp_path):
    from studio_agent.reference_assets import save_reference_upload

    p = _project(tmp_path)
    shots_path = p.path("storyboard", "shots.json")
    data = json.loads(shots_path.read_text())
    data["shots"][0]["reference_locations"] = ["Clock Shop"]
    shots_path.write_text(json.dumps(data))
    character = save_reference_upload(
        p,
        filename="mara.png",
        data=b"\x89PNG\r\n\x1a\nmara",
        target_type="character",
        target_id="Mara",
    )
    location = save_reference_upload(
        p,
        filename="shop.png",
        data=b"\x89PNG\r\n\x1a\nshop",
        target_type="location",
        target_id="Clock Shop",
    )

    class _RecordingAnalyzer(FakeReferenceAnalyzer):
        def __init__(self):
            self.seen = []

        def revise(self, image_paths, *, prompt, language="en"):
            self.seen.append(list(image_paths))
            return super().revise(image_paths, prompt=prompt, language=language)

    analyzer = _RecordingAnalyzer()
    revise_keyframe_from_feedback(
        p,
        "sh-001",
        "preserve the approved subjects",
        Providers(llm=FakeLLM(), reference_analyzer=analyzer),
    )

    assert str(p.dir / character["path"]) in analyzer.seen[0]
    assert str(p.dir / location["path"]) in analyzer.seen[0]


def test_keyframe_feedback_fails_before_revision_when_bound_upload_is_missing(tmp_path):
    from studio_agent.reference_assets import save_reference_upload

    p = _project(tmp_path)
    record = save_reference_upload(
        p,
        filename="mara.png",
        data=b"\x89PNG\r\n\x1a\nmissing",
        target_type="character",
        target_id="Mara",
    )
    p.path(*record["path"].split("/")).unlink()

    class _RecordingAnalyzer(FakeReferenceAnalyzer):
        def __init__(self):
            self.calls = []

        def revise(self, image_paths, *, prompt, language="en"):
            self.calls.append(list(image_paths))
            return super().revise(image_paths, prompt=prompt, language=language)

    analyzer = _RecordingAnalyzer()
    with pytest.raises(FileNotFoundError, match=record["path"]):
        revise_keyframe_from_feedback(
            p,
            "sh-001",
            "preserve Mara",
            Providers(llm=FakeLLM(), reference_analyzer=analyzer),
        )

    assert analyzer.calls == []


def test_video_feedback_persists_named_subject_reference_for_regeneration(tmp_path):
    from studio_agent.reference_assets import save_reference_upload

    p = _project(tmp_path)
    p.path("storyboard", "prompts", "sh-001.video.md").write_text("old video prompt")
    record = save_reference_upload(
        p,
        filename="hero.png",
        data=b"\x89PNG\r\n\x1a\nhero",
        target_type="character",
        target_id="Mara",
    )

    revise_video_from_feedback(
        p,
        "sh-001",
        "keep @image1 exactly",
        Providers(llm=FakeLLM()),
    )

    shot = json.loads(p.path("storyboard", "shots.json").read_text())["shots"][0]
    assert record["path"] in shot["reference_images"]
    assert shot["named_reference_images"] == [record["path"]]
