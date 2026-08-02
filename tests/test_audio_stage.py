"""Tests for legacy synthesis and provider-native audio extraction."""

import io
import json
import shutil
import subprocess
import wave

import pytest

from studio_agent.orchestrator.project import Project
from studio_agent.providers.fake import FakeMusic, FakeTTS
from studio_agent.stages.audio import AudioStage
from studio_agent.stages.base import Providers

PIPELINE = ["video", "audio"]

PLOT = {"themes": ["memory", "regret"], "logline": "A clockmaker who repairs memories."}
SHOTS = {"shots": [
    {"id": "sh-001", "duration_s": 3.0, "characters": ["Mara"],
     "dialogue": [{"character": "Mara", "line": "Another grey morning."}]},
    {"id": "sh-002", "duration_s": 2.0, "characters": [], "dialogue": []},  # no dialogue
]}


def _project(tmp_path):
    p = Project.create("audio test", root=tmp_path, stages=PIPELINE,
                       model_config={"style": {"look": "cinematic"}})
    p.path("story", "plot.json").write_text(json.dumps(PLOT))
    p.path("storyboard", "shots.json").write_text(json.dumps(SHOTS))
    return p


def _styled_project(tmp_path):
    p = Project.create(
        "styled audio test",
        root=tmp_path,
        stages=PIPELINE,
        model_config={"style": {
            "look": "cartoon",
            "palette": "bright candy colors",
            "rendering": "soft rounded shapes",
            "motion": "snappy squash-and-stretch energy",
        }},
    )
    p.path("story", "plot.json").write_text(json.dumps(PLOT))
    p.path("storyboard", "shots.json").write_text(json.dumps(SHOTS))
    return p


def _providers():
    return Providers(tts=FakeTTS(), music=FakeMusic())


def _silent_test_wav(duration_s=0.05, *, channels=2, frame_rate=48000):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(frame_rate)
        writer.writeframes(b"\x00\x00" * int(duration_s * frame_rate) * channels)
    return buffer.getvalue()


def _native_project(tmp_path, *, embedded_audio=False):
    p = Project.create(
        "native audio test",
        root=tmp_path,
        stages=PIPELINE,
        model_config={"audio_mode": "native_video"},
    )
    clip = p.path("assets", "clips", "sh-001.mp4")
    clip.write_bytes(b"FAKECLIP\nvideo")
    entry = {"id": "sh-001", "clip": clip.name, "duration_s": 3.0}
    if not embedded_audio:
        source = p.path("assets", "clips", "sh-001.native.source.wav")
        source.write_bytes(_silent_test_wav(3.0))
        entry["native_audio"] = source.name
    p.path("assets", "clips", "clips.json").write_text(
        json.dumps({"clips": [entry]})
    )
    return p


def _narrated_project(tmp_path):
    p = Project.create("narrated audio", root=tmp_path, stages=PIPELINE,
                       model_config={"style": {"look": "cinematic"}})
    p.path("story", "plot.json").write_text(json.dumps(PLOT))
    shots = {"shots": [
        {"id": "sh-001", "duration_s": 3.0, "characters": [], "dialogue": [],
         "narration": "Once, a clockmaker mended more than time."},
        {"id": "sh-002", "duration_s": 2.0, "characters": [], "dialogue": [],
         "narration": ""},
    ]}
    p.path("storyboard", "shots.json").write_text(json.dumps(shots))
    return p


def test_audio_synthesizes_narration_voiceover_for_narrated_shots(tmp_path):
    p = _narrated_project(tmp_path)

    result = AudioStage().run(p, _providers())

    assert result.status == "complete"
    assert p.path("assets", "audio", "sh-001.narration.wav").is_file()
    assert not p.path("assets", "audio", "sh-002.narration.wav").exists()
    # Editable narration prompt artifact (invariant #8).
    assert p.path("storyboard", "prompts", "sh-001.narration.md").is_file()
    manifest = json.loads(p.path("assets", "audio", "audio.json").read_text())
    assert [n["id"] for n in manifest["narration"]] == ["sh-001"]


def test_audio_writes_dialogue_only_for_shots_with_lines(tmp_path):
    p = _project(tmp_path)

    result = AudioStage().run(p, _providers())

    assert result.status == "complete"
    assert p.path("assets", "audio", "sh-001.dialogue.wav").is_file()
    assert not p.path("assets", "audio", "sh-002.dialogue.wav").exists()


def test_audio_writes_one_music_track_spanning_the_film(tmp_path):
    p = _project(tmp_path)
    p.model_config["music_enabled"] = True
    AudioStage().run(p, _providers())

    music = p.path("assets", "audio", "music.wav")
    assert music.is_file()
    manifest = json.loads(p.path("assets", "audio", "audio.json").read_text())
    # Music spans the total of all shot durations.
    assert manifest["music"]["duration_s"] == 5.0
    assert [d["id"] for d in manifest["dialogue"]] == ["sh-001"]


def test_audio_writes_editable_prompts(tmp_path):
    p = _project(tmp_path)
    p.model_config["music_enabled"] = True
    AudioStage().run(p, _providers())
    assert p.path("storyboard", "prompts", "sh-001.audio.md").is_file()
    assert p.path("storyboard", "prompts", "music.audio.md").is_file()


def test_legacy_music_only_when_enabled(tmp_path):
    p = _project(tmp_path)
    p.model_config["music_enabled"] = False
    AudioStage().run(p, _providers())
    manifest = json.loads(p.path("assets", "audio", "audio.json").read_text())
    assert not manifest.get("music")
    assert not p.path("assets", "audio", "music.wav").is_file()


def test_legacy_music_generated_when_enabled(tmp_path):
    p = _project(tmp_path)
    p.model_config["music_enabled"] = True
    p.model_config["music_mood"] = "sparse melancholic piano"
    AudioStage().run(p, _providers())
    manifest = json.loads(p.path("assets", "audio", "audio.json").read_text())
    assert manifest["music"]["file"] == "music.wav"
    assert p.path("assets", "audio", "music.wav").is_file()
    prompt = p.path("storyboard", "prompts", "music.audio.md").read_text()
    assert "sparse melancholic piano" in prompt


def test_music_prompt_includes_style_guidance(tmp_path):
    p = _styled_project(tmp_path)
    p.model_config["music_enabled"] = True

    AudioStage().run(p, _providers())

    prompt = p.path("storyboard", "prompts", "music.audio.md").read_text()
    assert "cartoon" in prompt
    assert "bright candy colors" in prompt
    assert "snappy squash-and-stretch energy" in prompt


def test_audio_logs_cost(tmp_path):
    p = _project(tmp_path)
    p.model_config["music_enabled"] = True
    AudioStage().run(p, _providers())
    # one dialogue line + one music track
    assert sum(1 for e in p.cost_log if e["stage"] == "audio") == 2


def test_audio_is_idempotent(tmp_path):
    p = _project(tmp_path)
    providers = _providers()
    AudioStage().run(p, providers)
    p.set_stage_status("audio", "complete")
    cost_after_first = len(p.cost_log)

    result = AudioStage().run(p, providers)
    assert result.status == "skipped"
    assert len(p.cost_log) == cost_after_first


def test_audio_regenerates_missing_dialogue_only(tmp_path):
    p = _project(tmp_path)
    providers = _providers()
    AudioStage().run(p, providers)
    cost_after_first = len(p.cost_log)

    p.path("assets", "audio", "sh-001.dialogue.wav").unlink()
    result = AudioStage().run(p, providers)

    assert result.status == "complete"
    assert p.path("assets", "audio", "sh-001.dialogue.wav").is_file()
    assert len(p.cost_log) == cost_after_first + 1  # only the one line re-rendered


def _native_narrated_project(tmp_path):
    p = Project.create(
        "native narrated", root=tmp_path, stages=PIPELINE,
        model_config={"audio_mode": "native_video"},
    )
    clip = p.path("assets", "clips", "sh-001.mp4")
    clip.write_bytes(b"FAKECLIP\nvideo")
    source = p.path("assets", "clips", "sh-001.native.source.wav")
    source.write_bytes(_silent_test_wav(3.0))
    p.path("assets", "clips", "clips.json").write_text(json.dumps(
        {"clips": [{"id": "sh-001", "clip": clip.name,
                    "native_audio": source.name, "duration_s": 3.0}]}))
    p.path("storyboard", "shots.json").write_text(json.dumps(
        {"shots": [{"id": "sh-001", "duration_s": 3.0,
                    "narration": "Once, a clockmaker mended more than time."}]}))
    return p


def test_native_audio_music_only_when_enabled(tmp_path):
    p = _native_project(tmp_path)
    p.model_config["music_enabled"] = False

    result = AudioStage().run(p, Providers(tts=FakeTTS(), music=FakeMusic()))

    assert result.status == "complete"
    manifest = json.loads(p.path("assets", "audio", "audio.json").read_text())
    assert not manifest.get("music")
    assert not p.path("assets", "audio", "music.wav").is_file()


def test_native_audio_music_generated_when_enabled(tmp_path):
    p = _native_project(tmp_path)
    p.model_config["music_enabled"] = True

    result = AudioStage().run(p, Providers(tts=FakeTTS(), music=FakeMusic()))

    assert result.status == "complete"
    manifest = json.loads(p.path("assets", "audio", "audio.json").read_text())
    assert manifest["music"]["file"] == "music.wav"
    assert p.path("assets", "audio", "music.wav").is_file()


def test_native_audio_synthesizes_narration_voiceover_stem(tmp_path):
    p = _native_narrated_project(tmp_path)

    result = AudioStage().run(p, Providers(tts=FakeTTS()))

    assert result.status == "complete"
    assert p.path("assets", "audio", "sh-001.narration.wav").is_file()
    manifest = json.loads(p.path("assets", "audio", "audio.json").read_text())
    assert manifest["mode"] == "native_video"
    assert [n["id"] for n in manifest["narration"]] == ["sh-001"]


def test_native_audio_uses_clip_sidecar_and_writes_manifest_without_provider_calls(tmp_path):
    p = _native_project(tmp_path)
    calls = []

    def extract(source, destination):
        calls.append((source, destination))
        destination.write_bytes(b"EXTRACTED")
        return destination

    result = AudioStage(extractor=extract).run(p, Providers())

    assert result.status == "complete"
    assert calls == [(
        p.path("assets", "clips", "sh-001.native.source.wav"),
        p.path("assets", "audio", "sh-001.native.wav"),
    )]
    manifest = json.loads(p.path("assets", "audio", "audio.json").read_text())
    assert manifest == {"mode": "native_video", "tracks": [{
        "id": "sh-001",
        "file": "sh-001.native.wav",
        "source_clip": "sh-001.mp4",
        "duration_s": 3.0,
    }]}
    assert "music" not in manifest
    assert p.cost_log == []


def test_native_audio_falls_back_to_embedded_clip_audio(tmp_path):
    p = _native_project(tmp_path, embedded_audio=True)
    sources = []

    def extract(source, destination):
        sources.append(source)
        destination.write_bytes(b"EXTRACTED")
        return destination

    result = AudioStage(extractor=extract).run(p, Providers())

    assert result.status == "complete"
    assert sources == [p.path("assets", "clips", "sh-001.mp4")]


def test_native_audio_preserves_hand_edited_wav(tmp_path):
    p = _native_project(tmp_path)
    calls = []

    def extract(source, destination):
        calls.append(source)
        destination.write_bytes(b"EXTRACTED")
        return destination

    stage = AudioStage(extractor=extract)
    stage.run(p, Providers())
    output = p.path("assets", "audio", "sh-001.native.wav")
    output.write_bytes(b"HAND EDIT")

    result = stage.run(p, Providers())

    assert result.status == "complete"
    assert output.read_bytes() == b"HAND EDIT"
    assert len(calls) == 1


def test_native_audio_extracts_editable_track_without_provider_calls(tmp_path):
    p = _native_project(tmp_path)

    result = AudioStage().run(p, Providers())

    assert result.status == "complete"
    output = p.path("assets", "audio", "sh-001.native.wav")
    assert output.is_file()
    with wave.open(str(output), "rb") as reader:
        assert reader.getnchannels() == 2
        assert reader.getframerate() == 48000
        assert reader.getsampwidth() == 2
        assert reader.getnframes() > 0
    assert p.cost_log == []


def test_native_audio_fails_instead_of_inserting_silence(tmp_path):
    p = _native_project(tmp_path)
    p.path("assets", "clips", "sh-001.native.source.wav").unlink()

    result = AudioStage().run(p, Providers())

    assert result.status == "failed"
    assert "sh-001" in result.message
    assert "audio" in result.message.lower()
    assert not p.path("assets", "audio", "sh-001.native.wav").exists()


def test_native_audio_fails_for_undecodable_sidecar(tmp_path):
    p = _native_project(tmp_path)
    p.path("assets", "clips", "sh-001.native.source.wav").write_bytes(b"not wav")

    result = AudioStage().run(p, Providers())

    assert result.status == "failed"
    assert "sh-001" in result.message
    assert "undecodable" in result.message


def test_native_audio_probe_accepts_valid_wav_and_rejects_undecodable_wav(tmp_path):
    from studio_agent.audio_native import has_audio_stream

    valid = tmp_path / "valid.wav"
    valid.write_bytes(_silent_test_wav())
    invalid = tmp_path / "invalid.wav"
    invalid.write_bytes(b"not a wave file")

    assert has_audio_stream(valid) is True
    assert has_audio_stream(invalid) is False


def test_extract_native_audio_normalizes_valid_wav(tmp_path):
    from studio_agent.audio_native import extract_native_audio

    source = tmp_path / "source.wav"
    source.write_bytes(_silent_test_wav())
    destination = tmp_path / "nested" / "output.wav"

    result = extract_native_audio(source, destination)

    assert result == destination
    with wave.open(str(destination), "rb") as reader:
        assert reader.getnchannels() == 2
        assert reader.getframerate() == 48000
        assert reader.getsampwidth() == 2
        assert reader.getcomptype() == "NONE"
        assert reader.getnframes() > 0


def test_extract_native_audio_rejects_truncated_wav(tmp_path):
    from studio_agent.audio_native import NativeAudioError, extract_native_audio

    source = tmp_path / "truncated.wav"
    source.write_bytes(_silent_test_wav()[:-100])
    destination = tmp_path / "output.wav"

    with pytest.raises(NativeAudioError, match="missing or undecodable"):
        extract_native_audio(source, destination)

    assert not destination.exists()


def test_extract_native_audio_normalizes_mono_44100_wav(tmp_path):
    from studio_agent.audio_native import extract_native_audio

    source = tmp_path / "mono-44100.wav"
    source.write_bytes(_silent_test_wav(channels=1, frame_rate=44100))
    destination = tmp_path / "normalized.wav"

    extract_native_audio(source, destination)

    with wave.open(str(destination), "rb") as reader:
        assert reader.getnchannels() == 2
        assert reader.getframerate() == 48000
        assert reader.getsampwidth() == 2
        assert reader.getcomptype() == "NONE"
        assert reader.getnframes() > 0


def test_extract_native_audio_accepts_ffmpeg_decodable_float_wav(tmp_path):
    from studio_agent.audio_native import extract_native_audio, has_audio_stream

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is required for native audio extraction")
    source = tmp_path / "float.wav"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=44100:cl=mono",
            "-t",
            "0.05",
            "-c:a",
            "pcm_f32le",
            str(source),
        ],
        check=True,
    )
    destination = tmp_path / "normalized.wav"

    assert has_audio_stream(source) is True
    assert extract_native_audio(source, destination) == destination
    with wave.open(str(destination), "rb") as reader:
        assert reader.getnchannels() == 2
        assert reader.getframerate() == 48000
        assert reader.getsampwidth() == 2
        assert reader.getcomptype() == "NONE"
        assert reader.getnframes() > 0


def test_failed_extraction_does_not_poison_retry(tmp_path, monkeypatch):
    from studio_agent import audio_native

    source = tmp_path / "source.wav"
    source.write_bytes(_silent_test_wav())
    destination = tmp_path / "output.wav"
    real_run = audio_native.subprocess.run

    def fail_after_partial_output(command, **kwargs):
        if command[0] == "ffmpeg":
            temp_output = command[-1]
            temp_output.write_bytes(b"partial wav")
            raise audio_native.subprocess.CalledProcessError(1, command)
        return real_run(command, **kwargs)

    monkeypatch.setattr(audio_native.subprocess, "run", fail_after_partial_output)
    with pytest.raises(audio_native.NativeAudioError, match="failed to extract"):
        audio_native.extract_native_audio(source, destination)

    assert not destination.exists()
    assert list(tmp_path.glob(".output.wav.*.tmp.wav")) == []

    monkeypatch.setattr(audio_native.subprocess, "run", real_run)
    assert audio_native.extract_native_audio(source, destination) == destination
    assert audio_native.has_audio_stream(destination)


def test_extract_native_audio_rejects_missing_or_undecodable_source(tmp_path):
    from studio_agent.audio_native import NativeAudioError, extract_native_audio

    for source in (tmp_path / "missing.wav", tmp_path / "invalid.wav"):
        if source.name == "invalid.wav":
            source.write_bytes(b"not a wave file")
        with pytest.raises(NativeAudioError, match="missing or undecodable"):
            extract_native_audio(source, tmp_path / "output.wav")
