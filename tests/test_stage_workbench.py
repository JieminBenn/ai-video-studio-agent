"""Stage Workbench context helpers for focused gate review."""

import json
import os

from studio_agent.orchestrator.project import Project
from studio_agent.stage_workbench import (
    _character_payloads,
    revision_target_for_stage,
    revision_targets_for_stage,
    stage_asset_actions,
    stage_payload,
    workbench_context,
)


PIPELINE = ["plot", "script", "bible", "storyboard", "video", "review", "audio", "assemble"]


def _project(tmp_path):
    p = Project.create("stage workbench", root=tmp_path, stages=PIPELINE)
    p.path("story", "plot.json").write_text(json.dumps({
        "logline": "A student wakes a dangerous power.",
        "synopsis": "He learns the cost of doing whatever he wants.",
        "characters": [{"name": "Mara", "role": "mentor", "description": "stern"}],
        "themes": ["power"],
    }))
    p.path("story", "script.json").write_text(json.dumps({
        "episodes": [{"scenes": [{
            "scene": 1,
            "heading": "INT. SCHOOL - NIGHT",
            "beats": ["The lockers bend inward."],
        }]}],
    }))
    cdir = p.path("bible", "characters", "mara")
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "character.json").write_text(json.dumps({"name": "Mara", "seed": 7}))
    (cdir / "identity_board.json").write_text(json.dumps({"canonical_face": "sharp"}))
    (cdir / "reference.png").write_bytes(b"ref")
    (cdir / "turnaround.png").write_bytes(b"turn")
    (cdir / "expressions.png").write_bytes(b"expr")
    (cdir / "states.json").write_text(json.dumps({
        "character": "Mara",
        "initial_state": "normal",
        "states": [
            {"id": "normal", "kind": "base", "description": "baseline"},
            {
                "id": "powered",
                "kind": "endpoint",
                "description": "glowing powered form",
                "reference_required": True,
                "reference_image": "states/powered/reference.png",
            },
        ],
        "transitions": [],
    }))
    state_dir = cdir / "states" / "powered"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "reference.png").write_bytes(b"powered")
    ldir = p.path("bible", "locations", "school")
    ldir.mkdir(parents=True, exist_ok=True)
    (ldir / "location.json").write_text(json.dumps({"name": "School"}))
    (ldir / "reference.png").write_bytes(b"loc-ref")
    p.path("bible", "style.md").write_text("# Style\n")
    p.path("storyboard", "shots.json").write_text(json.dumps({"shots": [{
        "id": "sh-001",
        "scene": 1,
        "description": "A hallway twists.",
        "camera": "wide",
        "action": "The student lifts one hand.",
        "duration_s": 3,
        "keyframe": "sh-001.png",
    }]}))
    p.path("storyboard", "prompts", "sh-001.keyframe.md").write_text("keyframe prompt")
    p.path("storyboard", "prompts", "sh-001.video.md").write_text("video prompt")
    p.path("storyboard", "prompts", "sh-001.audio.md").write_text("audio prompt")
    p.path("storyboard", "keyframes", "sh-001.png").write_bytes(b"kf")
    p.path("assets", "clips", "sh-001.mp4").write_bytes(b"clip")
    p.path("assets", "qc", "sh-001.json").write_text(json.dumps({"overall_pass": True}))
    p.path("edit", "timeline.json").write_text(json.dumps({"clips": ["sh-001"]}))
    return p


def test_workbench_context_focuses_current_stage_and_default_revision_target(tmp_path):
    p = _project(tmp_path)
    p.current_stage = "plot"

    ctx = workbench_context(p)

    assert ctx["stage"] == "plot"
    assert ctx["stage_label"] == "Plot"
    assert ctx["revision_target"]["path"] == "story/plot.json"
    assert "A student wakes a dangerous power." in ctx["summary"]


def test_workbench_summary_for_empty_plot_waits_for_output(tmp_path):
    p = Project.create("empty plot", root=tmp_path, stages=PIPELINE)

    ctx = workbench_context(p)

    assert ctx["stage"] == "plot"
    assert ctx["summary"] == "Waiting for plot output."


def test_revision_targets_follow_stage_defaults(tmp_path):
    p = _project(tmp_path)

    assert revision_target_for_stage(p, "plot")["path"] == "story/plot.json"
    assert revision_target_for_stage(p, "script")["path"] == "story/script.json"
    assert revision_target_for_stage(p, "storyboard")["path"] == "storyboard/shots.json"
    assert revision_target_for_stage(p, "video")["path"] == "storyboard/prompts/sh-001.video.md"
    assert revision_target_for_stage(p, "audio")["path"] == "storyboard/prompts/sh-001.audio.md"
    assert revision_target_for_stage(p, "assemble")["path"] == "edit/timeline.json"

    bible_targets = revision_targets_for_stage(p, "bible")
    assert [target["path"] for target in bible_targets[:3]] == [
        "bible/characters/mara/character.json",
        "bible/characters/mara/identity_board.json",
        "bible/characters/mara/states.json",
    ]


def test_script_stage_surfaces_structure_review_when_present(tmp_path):
    p = _project(tmp_path)
    p.path("story", "script.review.md").write_text("# 剧本结构审查 / Script structure review\n")

    paths = [target["path"] for target in revision_targets_for_stage(p, "script")]
    assert "story/script.review.md" in paths


def test_storyboard_stage_surfaces_flow_review_when_present(tmp_path):
    p = _project(tmp_path)
    p.path("storyboard", "shots.review.md").write_text("# 连贯性审查\n")

    paths = [target["path"] for target in revision_targets_for_stage(p, "storyboard")]
    assert "storyboard/shots.review.md" in paths
    assert "storyboard/shots.json" in paths


def test_bible_payload_exposes_state_rules_and_endpoint_images_for_manual_review(tmp_path):
    p = _project(tmp_path)

    character = stage_payload(p, "bible")["characters"][0]

    assert character["states"]["initial_state"] == "normal"
    assert {
        image["rel"] for image in character["images"]
    } >= {
        "bible/characters/mara/reference.png",
        "bible/characters/mara/states/powered/reference.png",
    }
    labels = {image["label"] for image in character["images"]}
    assert "Identity reference" in labels
    assert "Appearance state reference: powered" in labels


def test_native_audio_workbench_uses_video_sound_prompts_and_exposes_artifacts(tmp_path):
    p = _project(tmp_path)
    p.model_config["audio_mode"] = "native_video"
    p.path("assets", "audio", "sh-001.native.wav").write_bytes(b"wav")

    target = revision_target_for_stage(p, "audio")
    payload = stage_payload(p, "audio")
    shot = stage_payload(p, "review")["shots"][0]

    assert target == {
        "path": "storyboard/prompts/sh-001.video.md",
        "label": "Audiovisual Prompt: sh-001",
        "kind": "text",
    }
    assert payload["mode"] == "native_video"
    assert payload["prompts"] == ["storyboard/prompts/sh-001.video.md"]
    assert payload["source_clips"] == ["assets/clips/sh-001.mp4"]
    assert "qc_reports" not in payload
    assert shot["native_audio_rel"] == "assets/audio/sh-001.native.wav"


def test_legacy_audio_workbench_keeps_audio_prompt_and_hides_native_families(tmp_path):
    p = _project(tmp_path)
    p.model_config["audio_mode"] = "legacy"

    target = revision_target_for_stage(p, "audio")
    payload = stage_payload(p, "audio")

    assert target["path"] == "storyboard/prompts/sh-001.audio.md"
    assert payload == {
        "kind": "audio",
        "mode": "legacy",
        "prompts": ["storyboard/prompts/sh-001.audio.md"],
        "audio_files": [],
        "source_clips": [],
    }


def test_audio_workbench_filters_deleted_shot_prompts_but_keeps_legacy_music(tmp_path):
    p = _project(tmp_path)
    prompts = p.path("storyboard", "prompts")
    prompts.joinpath("deleted.video.md").write_text("stale video prompt")
    prompts.joinpath("deleted.audio.md").write_text("stale audio prompt")
    prompts.joinpath("music.audio.md").write_text("legacy music prompt")

    p.model_config["audio_mode"] = "native_video"
    native_targets = revision_targets_for_stage(p, "audio")
    native_payload = stage_payload(p, "audio")
    assert [target["path"] for target in native_targets] == [
        "storyboard/prompts/sh-001.video.md",
    ]
    assert native_payload["prompts"] == ["storyboard/prompts/sh-001.video.md"]

    p.model_config["audio_mode"] = "legacy"
    legacy_targets = revision_targets_for_stage(p, "audio")
    legacy_payload = stage_payload(p, "audio")
    assert [target["path"] for target in legacy_targets] == [
        "storyboard/prompts/music.audio.md",
        "storyboard/prompts/sh-001.audio.md",
    ]
    assert legacy_payload["prompts"] == [
        "storyboard/prompts/music.audio.md",
        "storyboard/prompts/sh-001.audio.md",
    ]


def test_storyboard_payload_exposes_keyframe_prompt_for_feedback_cards(tmp_path):
    p = _project(tmp_path)
    p.current_stage = "storyboard"

    ctx = workbench_context(p)
    shot = ctx["payload"]["shots"][0]

    assert shot["keyframe_rel"] == "storyboard/keyframes/sh-001.png"
    assert shot["keyframe_prompt_rel"] == "storyboard/prompts/sh-001.keyframe.md"


def test_clip_payload_exposes_keyframe_prompt_review_cards(tmp_path):
    p = _project(tmp_path)
    p.stages = ["concept", "bible", "clip", "video", "review", "assemble"]
    p.current_stage = "clip"

    ctx = workbench_context(p)
    shot = ctx["payload"]["shots"][0]

    assert ctx["stage"] == "clip"
    assert ctx["stage_label"] == "Clip"
    assert ctx["payload"]["kind"] == "keyframe_prompt_review"
    assert ctx["summary"] == "1 shot(s), 1 clip(s) on disk."
    assert ctx["revision_target"]["path"] == "storyboard/shots.json"
    assert shot["prompt_rel"] == "storyboard/prompts/sh-001.keyframe.md"


def test_stage_asset_actions_are_scoped_to_the_current_stage(tmp_path):
    p = _project(tmp_path)

    bible_actions = stage_asset_actions(p, "bible")
    assert {action["rel_path"] for action in bible_actions} == {
        "bible/characters/mara/reference.png",
        "bible/characters/mara/states/powered/reference.png",
        "bible/locations/school/reference.png",
    }

    keyframe_actions = stage_asset_actions(p, "keyframes")
    assert keyframe_actions == [{
        "kind": "keyframe",
        "shot_id": "sh-001",
        "rel_path": "storyboard/keyframes/sh-001.png",
        "label": "Regenerate keyframe sh-001",
    }]
    assert stage_asset_actions(p, "storyboard") == []
    assert stage_asset_actions(p, "clip") == []

    video_actions = stage_asset_actions(p, "video")
    assert video_actions == [{
        "kind": "shot_video",
        "shot_id": "sh-001",
        "rel_path": "assets/clips/sh-001.mp4",
        "label": "Regenerate video sh-001",
    }]

    # Automated LLM QC was removed: the review stage exposes no asset actions.
    assert stage_asset_actions(p, "review") == []


def test_workbench_exposes_pending_decision_and_stage_packet(tmp_path):
    p = _project(tmp_path)
    p.current_stage = "storyboard"
    decision = p.path("story", "decisions", "storyboard.json")
    decision.parent.mkdir(parents=True, exist_ok=True)
    decision.write_text(json.dumps({
        "stage": "storyboard",
        "question": "Should the camera observe or press closer?",
        "choices": [
            {"value": "observe", "label": "Observe"},
            {"value": "press closer", "label": "Press closer"},
        ],
        "default": "observe",
        "resolution": None,
    }))
    packet = p.path("knowledge", "packets", "shot-sh-001.json")
    packet.parent.mkdir(parents=True, exist_ok=True)
    packet.write_text(json.dumps({
        "purpose": "shot",
        "target": "sh-001",
        "selected_entries": [{
            "id": "camera.patient_push",
            "title": "Patient push-in",
            "reasons": ["intent"],
            "source": {"path": "core/foundations.yaml"},
        }],
    }))

    ctx = workbench_context(p)

    assert ctx["decision"]["question"].startswith("Should the camera")
    assert ctx["knowledge_packets"][0]["target"] == "sh-001"
    assert ctx["knowledge_packets"][0]["entries"][0]["title"] == "Patient push-in"


def test_stage_payload_style(tmp_path):
    from studio_agent.orchestrator.project import Project
    from studio_agent.stage_workbench import stage_payload
    p = Project.create("style wb", root=tmp_path, stages=["style", "bible"])
    p.path("bible", "style.md").write_text("# Show Bible — Visual Style\n\nmoody warm look\n")
    p.path("bible", "style_sample.png").write_bytes(b"\x89PNG\r\n")
    payload = stage_payload(p, "style")
    assert payload["kind"] == "style"
    assert "moody" in payload["style_md"]
    assert payload["sample_rel"] == "bible/style_sample.png"


def _prompt_gate_project(tmp_path, *, stage="clip", grid=False):
    stages = [stage, "keyframes", "video_prompts", "video"]
    project = Project.create("prompt gate workbench", root=tmp_path, stages=stages)
    shot = {
        "id": "sh-001",
        "scene": 1,
        "keyframe": "sh-001.png",
        "camera": "medium shot",
        "camera_movement": "slow push-in",
        "action": "Mara reaches for the clockwork bird",
        "start_frame": "Mara holds a still pose beside the clockwork bird",
        "reference_images": [],
    }
    if grid:
        shot["motion_grid"] = {
            "layout": "2x2", "rows": 2, "cols": 2, "panel_count": 4,
            "panel_beats": ["one", "two", "three", "four"],
        }
    project.path("storyboard", "shots.json").write_text(json.dumps({"shots": [shot]}))
    prompt_name = "sh-001.grid.md" if grid else "sh-001.keyframe.md"
    prompt = (
        "Panel 1 frozen still. Panel 2 frozen still. Panel 3 frozen still. "
        "Panel 4 frozen still."
        if grid else "STATIC EXACT PROMPT"
    )
    project.path("storyboard", "prompts", prompt_name).write_text(prompt)
    project.path("storyboard", "prompts", "sh-001.video.md").write_text(
        "VIDEO EXACT PROMPT"
    )
    return project


def test_clip_gate_payload_shows_exact_static_prompts_before_images(tmp_path):
    project = _prompt_gate_project(tmp_path, stage="clip")

    payload = stage_payload(project, "clip")

    assert payload["kind"] == "keyframe_prompt_review"
    assert payload["shots"][0]["prompt_text"] == "STATIC EXACT PROMPT"
    assert payload["shots"][0]["later_video_intent"]
    assert payload["shots"][0]["keyframe_rel"] == ""


def test_grid_gate_payload_uses_panel_plan_and_grid_prompt(tmp_path):
    project = _prompt_gate_project(tmp_path, stage="clip", grid=True)

    payload = stage_payload(project, "clip")

    assert payload["shots"][0]["prompt_kind"] == "grid"
    assert payload["shots"][0]["panel_count"] == 4
    assert "Panel 4" in payload["shots"][0]["prompt_text"]


def test_video_prompt_gate_payload_shows_keyframe_and_exact_motion_prompt(tmp_path):
    project = _prompt_gate_project(tmp_path, stage="clip")
    project.path("storyboard", "keyframes", "sh-001.png").write_bytes(b"keyframe")

    payload = stage_payload(project, "video_prompts")

    assert payload["kind"] == "video_prompt_review"
    assert payload["shots"][0]["keyframe_rel"]
    assert payload["shots"][0]["prompt_text"] == "VIDEO EXACT PROMPT"


def test_character_payload_text_ahead_tracks_mtime(tmp_path):
    project = _project(tmp_path)
    cdir = project.path("bible", "characters", "mara")

    # Deterministic False case: set reference.png mtime unambiguously newer than character.json.
    char_mtime = (cdir / "character.json").stat().st_mtime
    ref_newer = char_mtime + 10
    os.utime(cdir / "reference.png", (ref_newer, ref_newer))
    payload = _character_payloads(project)[0]
    assert payload["text_ahead"] is False

    # character.json newer than reference.png → ahead.
    later = ref_newer + 10
    os.utime(cdir / "character.json", (later, later))
    payload = _character_payloads(project)[0]
    assert payload["text_ahead"] is True


def test_character_payload_text_ahead_missing_reference_returns_true(tmp_path):
    project = _project(tmp_path)
    cdir = project.path("bible", "characters", "mara")

    # Missing reference.png → True (no image yet, text is ahead).
    (cdir / "reference.png").unlink()
    payload = _character_payloads(project)[0]
    assert payload["text_ahead"] is True


def test_character_payload_text_ahead_identity_board_newer_returns_true(tmp_path):
    project = _project(tmp_path)
    cdir = project.path("bible", "characters", "mara")

    # reference.png is the newest file.
    board_mtime = (cdir / "identity_board.json").stat().st_mtime
    ref_newest = board_mtime + 10
    os.utime(cdir / "reference.png", (ref_newest, ref_newest))
    assert _character_payloads(project)[0]["text_ahead"] is False

    # Now make identity_board.json newer → True.
    board_later = ref_newest + 10
    os.utime(cdir / "identity_board.json", (board_later, board_later))
    assert _character_payloads(project)[0]["text_ahead"] is True


def test_location_payload_exposes_text_ahead(tmp_path):
    from studio_agent.stage_workbench import _location_payloads

    project = _project(tmp_path)
    ldir = project.path("bible", "locations", "school")

    # reference.png is present (created by _project builder); set it newer than location.json.
    loc_mtime = (ldir / "location.json").stat().st_mtime
    ref_newer = loc_mtime + 10
    os.utime(ldir / "reference.png", (ref_newer, ref_newer))

    payload = _location_payloads(project)[0]
    assert "text_ahead" in payload
    assert isinstance(payload["text_ahead"], bool)
    assert payload["text_ahead"] is False

    # location.json newer → True.
    later = ref_newer + 10
    os.utime(ldir / "location.json", (later, later))
    assert _location_payloads(project)[0]["text_ahead"] is True
