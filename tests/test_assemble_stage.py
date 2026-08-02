"""Tests for the assemble stage — builds the EDL, then renders via an injected renderer.

Uses a FakeRenderer so the deterministic orchestration is tested without depending on
ffmpeg; the real ffmpeg render is verified by an end-to-end CLI run.
"""

import json

import pytest

from studio_agent.orchestrator.project import Project
from studio_agent.stages.assemble import AssembleStage
from studio_agent.stages.base import Providers

SHOTS = {"shots": [
    {"id": "sh-001", "duration_s": 3.0, "keyframe": "sh-001.png"},
    {"id": "sh-002", "duration_s": 2.0, "keyframe": "sh-002.png"},
]}
CLIPS = {"clips": [
    {"id": "sh-001", "clip": "sh-001.mp4", "duration_s": 3.0},
    {"id": "sh-002", "clip": "sh-002.mp4", "duration_s": 2.0},
]}
AUDIO = {"dialogue": [{"id": "sh-001", "file": "sh-001.dialogue.wav", "duration_s": 3.0}],
         "music": {"file": "music.wav", "duration_s": 5.0}}
NATIVE_AUDIO = {
    "mode": "native_video",
    "tracks": [
        {"id": "sh-001", "file": "sh-001.native.wav", "duration_s": 3.0},
        {"id": "sh-002", "file": "sh-002.native.wav", "duration_s": 2.0},
    ],
}


class FakeRenderer:
    def __init__(self):
        self.calls = []

    def render(self, project, timeline, out_path):
        self.calls.append(timeline)
        from pathlib import Path
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"FAKEMP4")
        return out_path


def _project(tmp_path):
    p = Project.create("assemble test", root=tmp_path, stages=["assemble"])
    p.path("storyboard", "shots.json").write_text(json.dumps(SHOTS))
    p.path("assets", "clips", "clips.json").write_text(json.dumps(CLIPS))
    p.path("assets", "audio", "audio.json").write_text(json.dumps(AUDIO))
    return p


def test_assemble_writes_timeline_and_output(tmp_path):
    p = _project(tmp_path)
    renderer = FakeRenderer()

    result = AssembleStage(renderer=renderer).run(p, Providers())

    assert result.status == "complete"
    assert p.path("edit", "timeline.json").is_file()
    out = p.path("output", f"{p.project_id}.mp4")
    assert out.is_file()
    assert len(renderer.calls) == 1


def test_assemble_passes_native_per_shot_audio_without_music_to_renderer(tmp_path):
    p = _project(tmp_path)
    p.model_config["audio_mode"] = "native_video"
    p.path("assets", "audio", "audio.json").write_text(json.dumps(NATIVE_AUDIO))
    renderer = FakeRenderer()

    result = AssembleStage(renderer=renderer).run(p, Providers())

    assert result.status == "complete"
    timeline = renderer.calls[0]
    assert timeline["audio_mode"] == "native_video"
    assert [segment["audio"] for segment in timeline["video_track"]] == [
        "sh-001.native.wav",
        "sh-002.native.wav",
    ]
    assert timeline["music"] is None


def test_assemble_rejects_missing_manifest_for_native_project(tmp_path):
    p = _project(tmp_path)
    p.model_config["audio_mode"] = "native_video"
    p.path("assets", "audio", "audio.json").unlink()

    with pytest.raises(RuntimeError, match="native audio manifest missing"):
        AssembleStage(renderer=FakeRenderer()).run(p, Providers())


def test_assemble_rejects_legacy_manifest_for_native_project(tmp_path):
    p = _project(tmp_path)
    p.model_config["audio_mode"] = "native_video"

    with pytest.raises(ValueError, match="must declare mode 'native_video'"):
        AssembleStage(renderer=FakeRenderer()).run(p, Providers())


def test_assemble_logs_cost(tmp_path):
    p = _project(tmp_path)
    AssembleStage(renderer=FakeRenderer()).run(p, Providers())
    assert any(e["stage"] == "assemble" for e in p.cost_log)


def test_assemble_is_idempotent_when_complete(tmp_path):
    p = _project(tmp_path)
    renderer = FakeRenderer()
    AssembleStage(renderer=renderer).run(p, Providers())
    p.set_stage_status("assemble", "complete")

    result = AssembleStage(renderer=renderer).run(p, Providers())
    assert result.status == "skipped"
    assert len(renderer.calls) == 1  # not re-rendered


def test_assemble_does_not_rerender_existing_output(tmp_path):
    p = _project(tmp_path)
    renderer = FakeRenderer()
    AssembleStage(renderer=renderer).run(p, Providers())

    # Re-run without marking complete: output already exists, so don't re-render.
    result = AssembleStage(renderer=renderer).run(p, Providers())
    assert result.status == "complete"
    assert len(renderer.calls) == 1
