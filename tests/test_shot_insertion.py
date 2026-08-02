"""User-driven shot insertion (add_shot) — both pipeline modes, offline on fakes."""

import json

import pytest

from studio_agent.orchestrator.project import Project
from studio_agent.prompt_approvals import confirm_prompt_batch
from studio_agent.providers.fake import FakeLLM
from studio_agent.shot_insertion import add_shot, next_shot_id
from studio_agent.stages.base import Providers


CLIP_PIPELINE = [
    "style", "concept", "bible", "clip", "keyframes", "video_prompts",
    "video", "audio", "assemble",
]
STORY_PIPELINE = [
    "style", "plot", "script", "bible", "storyboard", "keyframes",
    "video_prompts", "video", "audio", "assemble",
]


def _providers():
    return Providers(llm=FakeLLM())


def _clip_project(tmp_path):
    """Clip-mode project paused at the clip gate with two rendered clips."""
    p = Project.create("insert clip", root=tmp_path, stages=CLIP_PIPELINE)
    p.path("storyboard", "shots.json").write_text(json.dumps({"shots": [
        {"id": "sh-001", "scene": 1, "keyframe": "sh-001.png", "duration_s": 6,
         "deps": [], "continuity_reference": "", "keyframe_reference_images": [],
         "end_states": {"Hero": "human"}},
        {"id": "sh-002", "scene": 1, "keyframe": "sh-002.png", "duration_s": 6,
         "deps": ["sh-001"],
         "continuity_reference": "storyboard/keyframes/sh-001.png",
         "keyframe_reference_images": ["storyboard/keyframes/sh-001.png"]},
    ]}))
    p.path("storyboard", "visual_sequence.json").write_text(json.dumps({
        "version": 1,
        "target_duration_s": 12,
        "clip_durations": [6, 6],
        "beats": [{"id": "b1"}, {"id": "b2"}],
        "segments": [
            {"id": "segment-01", "duration_s": 6, "visual_beats": [{"id": "b1"}],
             "end_states": {"Hero": "human"}},
            {"id": "segment-02", "duration_s": 6, "visual_beats": [{"id": "b2"}]},
        ],
    }))
    for sid in ("sh-001", "sh-002"):
        p.path("storyboard", "keyframes", f"{sid}.png").write_bytes(f"kf {sid}".encode())
        p.path("storyboard", "prompts", f"{sid}.keyframe.md").write_text(f"prompt {sid}")
        p.path("assets", "clips", f"{sid}.mp4").write_bytes(f"clip {sid}".encode())
    p.path("assets", "clips", "clips.json").write_text(json.dumps({"clips": [
        {"id": "sh-001", "clip": "sh-001.mp4", "duration_s": 6},
        {"id": "sh-002", "clip": "sh-002.mp4", "duration_s": 6},
    ]}))
    p.path("output", f"{p.project_id}.mp4").write_bytes(b"final")
    for stage in ("style", "concept", "bible"):
        p.set_stage_status(stage, "approved")
    p.set_stage_status("clip", "complete")
    p.current_stage = "clip"
    p.status = "in_progress"
    p.save()
    return p


def _story_project(tmp_path):
    """Story-mode project with two scenes; sh-002 opens scene 2 and carries narration."""
    p = Project.create("insert story", root=tmp_path, stages=STORY_PIPELINE)
    p.path("storyboard", "shots.json").write_text(json.dumps({"shots": [
        {"id": "sh-001", "scene": 1, "keyframe": "sh-001.png", "duration_s": 3,
         "deps": [], "characters": ["Mara"]},
        {"id": "sh-002", "scene": 2, "keyframe": "sh-002.png", "duration_s": 3,
         "deps": ["sh-001"], "characters": ["Mara"], "narration": "夜幕降临。"},
        {"id": "sh-003", "scene": 2, "keyframe": "sh-003.png", "duration_s": 3,
         "deps": ["sh-002"], "characters": ["Mara"]},
    ]}))
    for sid in ("sh-001", "sh-002", "sh-003"):
        p.path("storyboard", "keyframes", f"{sid}.png").write_bytes(f"kf {sid}".encode())
        p.path("storyboard", "prompts", f"{sid}.keyframe.md").write_text(f"prompt {sid}")
    for stage in ("style", "plot", "script", "bible"):
        p.set_stage_status(stage, "approved")
    p.set_stage_status("storyboard", "complete")
    p.current_stage = "storyboard"
    p.status = "in_progress"
    p.save()
    return p


def test_next_shot_id_never_reuses_archived_ids(tmp_path):
    p = _clip_project(tmp_path)
    shots = json.loads(p.path("storyboard", "shots.json").read_text())["shots"]
    assert next_shot_id(p, shots) == "sh-003"
    # An archived sh-003 family (e.g. from a previous delete) blocks reuse.
    archived = p.path("history", "x-deleted-abc", "storyboard", "keyframes")
    archived.mkdir(parents=True)
    (archived / "sh-003.png").write_bytes(b"old")
    assert next_shot_id(p, shots) == "sh-004"


def test_add_shot_in_the_middle_rechains_clip_continuity(tmp_path):
    p = _clip_project(tmp_path)
    result = add_shot(
        p, description="插入一个特写：手中的灯笼亮起",
        after_shot_id="sh-001", providers=_providers(),
    )
    assert result.shot_id == "sh-003"
    assert result.position == 1
    assert result.planning_stage == "clip"

    shots = json.loads(p.path("storyboard", "shots.json").read_text())["shots"]
    assert [s["id"] for s in shots] == ["sh-001", "sh-003", "sh-002"]
    # The fake echoes the request verbatim — no language interpretation.
    assert shots[1]["description"] == "插入一个特写：手中的灯笼亮起"
    assert shots[1]["deps"] == ["sh-001"]
    assert shots[2]["deps"] == ["sh-003"]
    # Successor's clip continuity now anchors on the inserted shot's keyframe.
    assert shots[2]["continuity_reference"] == "storyboard/keyframes/sh-003.png"
    # The inserted beat inherits the previous segment's end state (no new transformation).
    assert shots[1]["start_states"] == {"Hero": "human"}

    sequence = json.loads(p.path("storyboard", "visual_sequence.json").read_text())
    assert len(sequence["segments"]) == 3
    assert sequence["clip_durations"][1] == shots[1]["duration_s"]

    # Pipeline re-pended from the planning stage so the resume drafts the new prompt.
    reloaded = Project.load(p.dir)
    assert reloaded.current_stage == "clip"
    assert reloaded.stage_status("clip") == "pending"
    assert reloaded.stage_status("video") == "pending"


def test_add_shot_at_start_rebuilds_the_chain_head(tmp_path):
    p = _clip_project(tmp_path)
    result = add_shot(
        p, description="an establishing wide of the empty street",
        after_shot_id="", providers=_providers(),
    )
    shots = json.loads(p.path("storyboard", "shots.json").read_text())["shots"]
    assert [s["id"] for s in shots] == [result.shot_id, "sh-001", "sh-002"]
    assert shots[0]["deps"] == []
    assert shots[1]["deps"] == [result.shot_id]
    assert shots[1]["continuity_reference"] == f"storyboard/keyframes/{result.shot_id}.png"


def test_add_shot_never_touches_existing_artifacts_but_archives_aggregates(tmp_path):
    p = _clip_project(tmp_path)
    add_shot(p, description="a new beat", after_shot_id="sh-001", providers=_providers())
    # Existing paid artifacts stay live (localized regeneration).
    for sid in ("sh-001", "sh-002"):
        assert p.path("storyboard", "keyframes", f"{sid}.png").is_file()
        assert p.path("assets", "clips", f"{sid}.mp4").is_file()
        assert p.path("storyboard", "prompts", f"{sid}.keyframe.md").is_file()
    # Whole-film aggregates are archived so the re-pended stages rebuild them.
    assert not p.path("assets", "clips", "clips.json").is_file()
    assert not p.path("output", f"{p.project_id}.mp4").is_file()
    assert list(p.dir.joinpath("history").rglob("clips.json"))


def test_add_shot_rebases_prompt_approvals_dropping_new_and_successor(tmp_path):
    p = _clip_project(tmp_path)
    confirm_prompt_batch(p, "keyframes", confirmer="test")
    add_shot(p, description="a new beat", after_shot_id="sh-001", providers=_providers())
    manifest = json.loads(
        p.path("storyboard", "prompt_approvals", "keyframes.json").read_text()
    )
    assert "sh-001" in manifest["prompts"]      # untouched approval survives
    assert "sh-002" not in manifest["prompts"]  # successor re-reviews (refs changed)
    assert "sh-003" not in manifest["prompts"]  # the new shot has no approval yet


def test_add_shot_story_mode_inherits_scene_without_clip_continuity(tmp_path):
    p = _story_project(tmp_path)
    result = add_shot(
        p, description="Mara pauses at the window", after_shot_id="sh-002",
        providers=_providers(),
    )
    shots = json.loads(p.path("storyboard", "shots.json").read_text())["shots"]
    assert [s["id"] for s in shots] == ["sh-001", "sh-002", result.shot_id, "sh-003"]
    inserted = shots[2]
    assert inserted["scene"] == 2                       # neighbor's scene
    assert not inserted.get("continuity_reference")     # clip-mode field never injected
    assert "visual_beats" not in inserted
    assert inserted["narration"] == ""
    assert inserted["duration_s"] == pytest.approx(4.0)  # story mode keeps LLM seconds
    assert shots[3]["deps"] == [result.shot_id]
    assert result.planning_stage == "storyboard"
    # Existing shots keep story-mode purity: no clip continuity fields appeared.
    assert not shots[3].get("continuity_reference")


def test_add_shot_validation_errors(tmp_path):
    p = _clip_project(tmp_path)
    with pytest.raises(ValueError):
        add_shot(p, description="   ", after_shot_id="sh-001", providers=_providers())
    with pytest.raises(FileNotFoundError):
        add_shot(p, description="x", after_shot_id="sh-999", providers=_providers())
    with pytest.raises(ValueError):
        add_shot(p, description="x", after_shot_id="", providers=Providers(llm=None))


def test_delete_story_shot_carries_narration_to_same_scene_neighbor(tmp_path):
    from studio_agent.asset_regeneration import delete_clip_shot

    p = _story_project(tmp_path)
    p.path("assets", "audio", "sh-003.narration.wav").write_bytes(b"stale")
    p.path("assets", "audio", "audio.json").write_text(json.dumps({"narration": []}))
    delete_clip_shot(p, "sh-002")

    shots = json.loads(p.path("storyboard", "shots.json").read_text())["shots"]
    assert [s["id"] for s in shots] == ["sh-001", "sh-003"]
    # 旁白 moved verbatim to the surviving shot of scene 2 (invariant #11: field move only).
    assert shots[1]["narration"] == "夜幕降临。"
    # Survivors gained no clip-mode continuity fields.
    assert not shots[0].get("continuity_reference")
    assert not shots[1].get("continuity_reference")
    assert shots[1]["deps"] == ["sh-001"]
    # The receiver's stale narration stem re-synthesizes.
    assert not p.path("assets", "audio", "sh-003.narration.wav").is_file()
    assert not p.path("assets", "audio", "audio.json").is_file()
