"""CLI regeneration tests for localized shot refresh."""

import json
from pathlib import Path

from studio_agent import cli
from studio_agent.providers.fake import (
    FakeImageGen,
    FakeMusic,
    FakeTTS,
    FakeVideoGen,
    FakeVLMCheck,
)
from studio_agent.orchestrator.project import Project
from studio_agent.prompt_approvals import confirm_prompt_batch
from studio_agent.stages.assemble import AssembleStage
from studio_agent.stages.audio import AudioStage
from studio_agent.stages.base import Providers
from studio_agent.stages.review import ReviewStage
from studio_agent.stages.video import VideoStage
from studio_agent.stages.video_prompts import VideoPromptsStage


SHOTS = {
    "shots": [
        {
            "id": "sh-001",
            "scene": 1,
            "description": "wide",
            "camera": "wide",
            "action": "open",
            "dialogue": [{"character": "Mara", "line": "Hello again."}],
            "duration_s": 2.0,
            "characters": ["Mara"],
            "deps": [],
            "reference_seed": 11,
            "keyframe": "sh-001.png",
        },
        {
            "id": "sh-002",
            "scene": 1,
            "description": "close",
            "camera": "close-up",
            "action": "react",
            "dialogue": [{"character": "Mara", "line": "Still here."}],
            "duration_s": 2.0,
            "characters": ["Mara"],
            "deps": ["sh-001"],
            "reference_seed": 11,
            "keyframe": "sh-002.png",
        },
    ]
}


class FakeRenderer:
    def render(self, project, timeline, out_path):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(json.dumps(timeline, sort_keys=True).encode())
        return out_path


def _providers():
    return Providers(video=FakeVideoGen(), vlm=FakeVLMCheck(), tts=FakeTTS(), music=FakeMusic())


def _project(tmp_path, *, native_audio=False):
    model_config = {"video": "fake", "vlm": "fake", "tts": "fake", "music": "fake"}
    if native_audio:
        model_config["audio_mode"] = "native_video"
    p = Project.create(
        "cli regen test",
        root=tmp_path,
        stages=["video_prompts", "video", "review", "audio", "assemble"],
        model_config=model_config,
    )
    p.path("storyboard", "shots.json").write_text(json.dumps(SHOTS))
    p.path("story", "plot.json").write_text(json.dumps({"logline": "test", "themes": []}))
    for shot in SHOTS["shots"]:
        FakeImageGen().generate(
            "kf",
            out_path=str(p.path("storyboard", "keyframes", shot["keyframe"])),
            seed=shot["reference_seed"],
        )

    renderer = FakeRenderer()
    VideoPromptsStage().run(p, _providers())
    confirm_prompt_batch(p, "videos", confirmer="test")
    VideoStage().run(p, _providers())
    ReviewStage().run(p, _providers())
    AudioStage().run(p, _providers())
    AssembleStage(renderer=renderer).run(p, _providers())
    for stage in p.stages:
        p.set_stage_status(stage, "approved")
    p.current_stage = None
    p.status = "done"
    p.save()
    return p


def test_cli_regenerate_shot_refreshes_only_requested_shot(monkeypatch, tmp_path):
    p = _project(tmp_path)
    monkeypatch.setattr(cli, "PROJECTS_ROOT", tmp_path)
    monkeypatch.setattr(
        cli,
        "STAGES",
        [VideoStage(), ReviewStage(), AudioStage(), AssembleStage(renderer=FakeRenderer())],
    )

    p.path("storyboard", "prompts", "sh-001.video.md").write_text(
        "human edited video prompt: Mara opens the door. Beat 1 establishes her, then "
        "Beat 2 resolves the action. A gentle push-in follows the opening movement. "
        "50mm lens at eye-level with motivated window light. Exact dialogue: Hello again."
    )
    untouched_clip = p.path("assets", "clips", "sh-002.mp4").read_bytes()
    cost_before = len(p.cost_log)

    code = cli.main(["regenerate", p.project_id, "--shot", "sh-001"])

    assert code == 0
    refreshed = p.path("assets", "clips", "sh-001.mp4").read_bytes()
    assert b"human edited video prompt" in refreshed
    assert p.path("assets", "clips", "sh-002.mp4").read_bytes() == untouched_clip
    assert p.path("assets", "audio", "sh-001.dialogue.wav").is_file()
    assert p.path("assets", "qc", "sh-001.json").is_file()
    assert p.path("output", f"{p.project_id}.mp4").is_file()
    archived_clips = list(p.path("history").rglob("assets/clips/sh-001.mp4"))
    assert len(archived_clips) == 1

    reloaded = Project.load(p.dir)
    assert reloaded.status == "done"
    assert reloaded.current_stage is None
    assert all(reloaded.stage_status(s) == "approved" for s in p.stages)
    assert len(reloaded.cost_log) == cost_before + 4


def test_cli_regenerate_native_project_replaces_only_selected_wav(monkeypatch, tmp_path):
    p = _project(tmp_path, native_audio=True)
    monkeypatch.setattr(cli, "PROJECTS_ROOT", tmp_path)
    monkeypatch.setattr(
        cli,
        "STAGES",
        [VideoStage(), ReviewStage(), AudioStage(), AssembleStage(renderer=FakeRenderer())],
    )

    selected_audio = p.path("assets", "audio", "sh-001.native.wav")
    selected_audio.write_bytes(b"stale selected native audio")
    untouched_audio = p.path("assets", "audio", "sh-002.native.wav").read_bytes()
    cost_before = len(p.cost_log)

    code = cli.main(["regenerate", p.project_id, "--shot", "sh-001"])

    assert code == 0
    assert selected_audio.is_file()
    assert selected_audio.read_bytes() != b"stale selected native audio"
    assert p.path("assets", "audio", "sh-002.native.wav").read_bytes() == untouched_audio
    assert len(list(p.path("history").rglob("assets/audio/sh-001.native.wav"))) == 1
    assert not list(p.path("history").rglob("assets/audio/sh-002.native.wav"))
    assert not p.path("assets", "audio", "music.wav").exists()

    reloaded = Project.load(p.dir)
    assert reloaded.status == "done"
    assert len(reloaded.cost_log) == cost_before + 3


def test_cli_answer_decision_resolves_file(monkeypatch, tmp_path):
    p = Project.create("decision cli", root=tmp_path, stages=["bible"])
    path = p.path("story", "decisions", "bible.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "stage": "bible",
        "question": "Should Mara feel carefully composed or visibly worn down?",
        "choices": [
            {"value": "carefully composed", "label": "Carefully composed"},
            {"value": "visibly worn down", "label": "Visibly worn down"},
        ],
        "default": "carefully composed",
        "resolution": None,
    }))
    monkeypatch.setattr(cli, "PROJECTS_ROOT", tmp_path)

    assert cli.main([
        "answer-decision",
        p.project_id,
        "--stage",
        "bible",
        "--choice",
        "visibly worn down",
    ]) == 0

    data = json.loads(path.read_text())
    assert data["resolution"]["value"] == "visibly worn down"
    assert data["resolution"]["source"] == "user"


def test_cli_knowledge_import_and_rebuild_index(monkeypatch, tmp_path, capsys):
    source = tmp_path / "camera-notes.md"
    source.write_text("# Patient push-in\n\nMove only when the emotional distance changes.\n")
    knowledge_root = tmp_path / "knowledge"
    monkeypatch.setattr(cli, "KNOWLEDGE_ROOT", knowledge_root, raising=False)

    assert cli.main([
        "knowledge", "import", str(source),
        "--domain", "camera_movement",
        "--stage", "storyboard",
        "--language", "en",
    ]) == 0
    output = capsys.readouterr().out
    assert "chunks: 1" in output
    assert "manifest" in output

    assert cli.main(["knowledge", "rebuild-index"]) == 0
    assert knowledge_root.joinpath("index.json").is_file()


def test_cli_refreshes_one_shot_packet(monkeypatch, tmp_path):
    p = _project(tmp_path)
    packet = p.path("knowledge", "packets", "shot-sh-001.json")
    packet.parent.mkdir(parents=True, exist_ok=True)
    packet.write_text(json.dumps({"purpose": "shot", "target": "sh-001"}))
    untouched = p.path("assets", "clips", "sh-002.mp4").read_bytes()
    monkeypatch.setattr(cli, "PROJECTS_ROOT", tmp_path)

    assert cli.main([
        "knowledge", "refresh", p.project_id,
        "--purpose", "shot",
        "--target", "sh-001",
    ]) == 0

    assert not packet.exists()
    assert p.path("assets", "clips", "sh-002.mp4").read_bytes() == untouched
