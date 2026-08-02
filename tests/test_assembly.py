"""Tests for deterministic EDL building and FFmpeg renderer timing."""

import json
import shutil
import struct
import subprocess
import wave

import pytest

from studio_agent.assembly import ffmpeg_edit
from studio_agent.assembly.ffmpeg_edit import FFmpegRenderer, build_timeline
from studio_agent.orchestrator.project import Project
from studio_agent.providers.fake import FakeImageGen

SHOTS = {"shots": [
    {"id": "sh-001", "duration_s": 3.0, "keyframe": "sh-001.png"},
    {"id": "sh-002", "duration_s": 2.0, "keyframe": "sh-002.png"},
]}
CLIPS = {"clips": [
    {"id": "sh-001", "clip": "sh-001.mp4", "duration_s": 3.0},
    {"id": "sh-002", "clip": "sh-002.mp4", "duration_s": 2.0},
]}
AUDIO = {
    "dialogue": [{"id": "sh-001", "file": "sh-001.dialogue.wav", "duration_s": 3.0}],
    "music": {"file": "music.wav", "duration_s": 5.0},
}
AUDIO_WITH_NARRATION = {
    "dialogue": [{"id": "sh-001", "file": "sh-001.dialogue.wav", "duration_s": 3.0}],
    "narration": [{"id": "sh-002", "file": "sh-002.narration.wav", "duration_s": 2.0}],
    "music": {"file": "music.wav", "duration_s": 5.0},
}
NATIVE_AUDIO = {
    "mode": "native_video",
    "tracks": [
        {
            "id": "sh-001",
            "file": "sh-001.native.wav",
            "source_clip": "sh-001.mp4",
            "duration_s": 3.0,
        },
        {
            "id": "sh-002",
            "file": "sh-002.native.wav",
            "source_clip": "sh-002.mp4",
            "duration_s": 2.0,
        },
    ],
}


def _project(tmp_path):
    p = Project.create("edl test", root=tmp_path, stages=["assemble"])
    p.path("storyboard", "shots.json").write_text(json.dumps(SHOTS))
    p.path("assets", "clips", "clips.json").write_text(json.dumps(CLIPS))
    p.path("assets", "audio", "audio.json").write_text(json.dumps(AUDIO))
    return p


def test_build_timeline_orders_shots_and_sums_duration(tmp_path):
    p = _project(tmp_path)

    timeline = build_timeline(p)

    assert [seg["id"] for seg in timeline["video_track"]] == ["sh-001", "sh-002"]
    assert timeline["total_duration_s"] == 5.0
    assert timeline["video_track"][0]["keyframe"] == "sh-001.png"
    assert timeline["video_track"][0]["duration_s"] == 3.0


def test_build_timeline_maps_dialogue_per_shot(tmp_path):
    p = _project(tmp_path)
    timeline = build_timeline(p)

    seg1, seg2 = timeline["video_track"]
    assert seg1["dialogue"] == "sh-001.dialogue.wav"  # shot with a line
    assert seg2["dialogue"] is None                   # shot without
    assert timeline["music"] == "music.wav"


def test_build_timeline_maps_narration_per_shot(tmp_path):
    p = _project(tmp_path)
    p.path("assets", "audio", "audio.json").write_text(json.dumps(AUDIO_WITH_NARRATION))

    timeline = build_timeline(p)

    seg1, seg2 = timeline["video_track"]
    assert seg1["narration"] is None                       # shot without narration
    assert seg2["narration"] == "sh-002.narration.wav"     # narrated shot


def test_legacy_renderer_builds_and_muxes_narration_stem(tmp_path):
    p = _project(tmp_path)
    p.path("assets", "audio", "audio.json").write_text(json.dumps(AUDIO_WITH_NARRATION))
    for shot in SHOTS["shots"]:
        p.path("storyboard", "keyframes", shot["keyframe"]).write_bytes(b"png")
    p.path("assets", "audio", "sh-001.dialogue.wav").write_bytes(b"dialogue")
    p.path("assets", "audio", "sh-002.narration.wav").write_bytes(b"narration")

    renderer = RecordingRenderer()
    renderer.render(p, build_timeline(p), str(p.path("output", "narrated.mp4")))

    # Narrated shot's VO is fit to its shot duration; the other shot is silent VO.
    assert ("sh-002.narration.wav", 2.0) in renderer.audio_fits
    # Production (dialogue) track is unchanged; a dedicated narration stem is built.
    assert renderer.audio_concats[0] == [b"dialogue", b"silence"]
    assert renderer.audio_concats[1] == [b"silence", b"narration"]
    # The narration stem reaches the mux.
    assert renderer.mux_narration is not None


def test_legacy_renderer_without_narration_passes_none_to_mux(tmp_path):
    p = _project(tmp_path)  # default AUDIO has no narration
    for shot in SHOTS["shots"]:
        p.path("storyboard", "keyframes", shot["keyframe"]).write_bytes(b"png")
    p.path("assets", "audio", "sh-001.dialogue.wav").write_bytes(b"dialogue")

    renderer = RecordingRenderer()
    renderer.render(p, build_timeline(p), str(p.path("output", "plain.mp4")))

    assert renderer.mux_narration is None
    assert len(renderer.audio_concats) == 1  # production track only


def test_native_timeline_maps_tracks_and_has_no_music(tmp_path):
    p = _project(tmp_path)
    p.model_config["audio_mode"] = "native_video"
    p.path("assets", "audio", "audio.json").write_text(json.dumps(NATIVE_AUDIO))

    timeline = build_timeline(p)

    assert timeline["audio_mode"] == "native_video"
    assert [segment["audio"] for segment in timeline["video_track"]] == [
        "sh-001.native.wav",
        "sh-002.native.wav",
    ]
    assert all("dialogue" not in segment for segment in timeline["video_track"])
    assert timeline["music"] is None


def test_native_timeline_maps_narration_per_shot(tmp_path):
    p = _project(tmp_path)
    p.model_config["audio_mode"] = "native_video"
    manifest = json.loads(json.dumps(NATIVE_AUDIO))
    manifest["narration"] = [
        {"id": "sh-002", "file": "sh-002.narration.wav", "duration_s": 2.0}
    ]
    p.path("assets", "audio", "audio.json").write_text(json.dumps(manifest))

    timeline = build_timeline(p)

    seg1, seg2 = timeline["video_track"]
    assert seg1["narration"] is None
    assert seg2["narration"] == "sh-002.narration.wav"


def test_native_timeline_surfaces_muxed_music_when_present(tmp_path):
    p = _project(tmp_path)
    p.model_config["audio_mode"] = "native_video"
    manifest = json.loads(json.dumps(NATIVE_AUDIO))
    manifest["music"] = {"file": "music.wav", "duration_s": 5.0}
    p.path("assets", "audio", "audio.json").write_text(json.dumps(manifest))

    timeline = build_timeline(p)

    assert timeline["music"] == "music.wav"


def test_native_timeline_has_no_music_key_value_when_absent(tmp_path):
    p = _project(tmp_path)
    p.model_config["audio_mode"] = "native_video"
    p.path("assets", "audio", "audio.json").write_text(json.dumps(NATIVE_AUDIO))

    timeline = build_timeline(p)

    assert timeline.get("music") is None


def test_native_renderer_mixes_narration_stem_over_production_track(tmp_path):
    p = _project(tmp_path)
    p.model_config["audio_mode"] = "native_video"
    manifest = json.loads(json.dumps(NATIVE_AUDIO))
    manifest["narration"] = [
        {"id": "sh-002", "file": "sh-002.narration.wav", "duration_s": 2.0}
    ]
    p.path("assets", "audio", "audio.json").write_text(json.dumps(manifest))
    for shot in SHOTS["shots"]:
        p.path("storyboard", "keyframes", shot["keyframe"]).write_bytes(b"png")
    p.path("assets", "audio", "sh-001.native.wav").write_bytes(b"native-one")
    p.path("assets", "audio", "sh-002.native.wav").write_bytes(b"native-two")
    p.path("assets", "audio", "sh-002.narration.wav").write_bytes(b"narration")

    renderer = RecordingRenderer()
    renderer.render(p, build_timeline(p), str(p.path("output", "native_vo.mp4")))

    assert renderer.audio_concats[0] == [b"native-one", b"native-two"]  # production
    assert renderer.audio_concats[1] == [b"silence", b"narration"]      # narration stem
    assert renderer.mux_narration is not None
    assert renderer.mux_music is None  # manifest carries no music entry here


def test_native_project_rejects_missing_audio_manifest(tmp_path):
    p = _project(tmp_path)
    p.model_config["audio_mode"] = "native_video"
    p.path("assets", "audio", "audio.json").unlink()

    with pytest.raises(RuntimeError, match="native audio manifest missing"):
        build_timeline(p)


@pytest.mark.parametrize("manifest_mode", [None, "legacy", "future_mode"])
def test_native_project_rejects_mismatched_audio_manifest_mode(tmp_path, manifest_mode):
    p = _project(tmp_path)
    p.model_config["audio_mode"] = "native_video"
    manifest = json.loads(json.dumps(NATIVE_AUDIO))
    if manifest_mode is None:
        manifest.pop("mode")
    else:
        manifest["mode"] = manifest_mode
    p.path("assets", "audio", "audio.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="must declare mode 'native_video'"):
        build_timeline(p)


def test_native_project_rejects_malformed_audio_manifest_json(tmp_path):
    p = _project(tmp_path)
    p.model_config["audio_mode"] = "native_video"
    p.path("assets", "audio", "audio.json").write_text("{not-json")

    with pytest.raises(ValueError, match="invalid native audio manifest JSON"):
        build_timeline(p)


def test_native_project_rejects_manifest_missing_a_shot_track(tmp_path):
    p = _project(tmp_path)
    p.model_config["audio_mode"] = "native_video"
    manifest = json.loads(json.dumps(NATIVE_AUDIO))
    manifest["tracks"] = manifest["tracks"][:1]
    p.path("assets", "audio", "audio.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="native audio manifest missing track for shot sh-002"):
        build_timeline(p)


def test_project_rejects_unknown_explicit_audio_mode(tmp_path):
    p = _project(tmp_path)
    p.model_config["audio_mode"] = "future_mode"

    with pytest.raises(ValueError, match="unsupported project audio_mode"):
        build_timeline(p)


def test_build_timeline_persists_to_edit_folder(tmp_path):
    p = _project(tmp_path)
    build_timeline(p)

    saved = json.loads(p.path("edit", "timeline.json").read_text())
    assert saved["total_duration_s"] == 5.0


class RecordingRenderer(FFmpegRenderer):
    def __init__(self):
        self.video_sources = []
        self.audio_fits = []
        self.audio_concats = []
        self.mux_music = object()
        self.mux_narration = object()

    @property
    def audio_sources(self):
        # The production (dialogue/native) track is always the first audio concat.
        return self.audio_concats[0] if self.audio_concats else []

    def _clip_to_segment(self, clip, duration, out):
        self.video_sources.append(("clip", clip.name, duration))
        out.write_bytes(b"clip-segment")

    def _image_to_segment(self, image, duration, out):
        self.video_sources.append(("image", image.name, duration))
        out.write_bytes(b"image-segment")

    def _silence(self, duration, out):
        out.write_bytes(b"silence")

    def _fit_audio_segment(self, source, duration, out):
        self.audio_fits.append((source.name, duration))
        out.write_bytes(source.read_bytes())

    def _concat(self, parts, out, *, kind):
        if kind == "a":
            self.audio_concats.append([part.read_bytes() for part in parts])
        out.write_bytes(kind.encode("ascii"))

    def _mux(self, video, production_audio, narration, music, out):
        self.mux_narration = narration
        self.mux_music = music
        out.write_bytes(b"muxed")


def test_renderer_uses_real_mp4_clips_before_keyframe_fallback(tmp_path):
    p = _project(tmp_path)
    for shot in SHOTS["shots"]:
        p.path("storyboard", "keyframes", shot["keyframe"]).write_bytes(b"png")
    p.path("assets", "clips", "sh-001.mp4").write_bytes(
        b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"
    )
    p.path("assets", "clips", "sh-002.mp4").write_bytes(b"FAKECLIP\nplaceholder")

    timeline = build_timeline(p)
    renderer = RecordingRenderer()
    renderer.render(p, timeline, str(p.path("output", "out.mp4")))

    assert renderer.video_sources == [
        ("clip", "sh-001.mp4", 3.0),
        ("image", "sh-002.png", 2.0),
    ]


def test_legacy_renderer_also_fits_dialogue_without_changing_silent_shots(tmp_path):
    p = _project(tmp_path)
    for shot in SHOTS["shots"]:
        p.path("storyboard", "keyframes", shot["keyframe"]).write_bytes(b"png")
    p.path("assets", "audio", "sh-001.dialogue.wav").write_bytes(b"dialogue")

    renderer = RecordingRenderer()
    renderer.render(p, build_timeline(p), str(p.path("output", "legacy.mp4")))

    assert renderer.audio_fits == [("sh-001.dialogue.wav", 3.0)]
    assert renderer.audio_sources == [b"dialogue", b"silence"]


def test_native_renderer_concatenates_each_shot_track_once_and_mixes_score_when_present(tmp_path):
    p = _project(tmp_path)
    p.model_config["audio_mode"] = "native_video"
    p.path("assets", "audio", "audio.json").write_text(json.dumps(NATIVE_AUDIO))
    for shot in SHOTS["shots"]:
        p.path("storyboard", "keyframes", shot["keyframe"]).write_bytes(b"png")
    p.path("assets", "audio", "sh-001.native.wav").write_bytes(b"native-one")
    p.path("assets", "audio", "sh-002.native.wav").write_bytes(b"native-two")
    p.path("assets", "audio", "score.wav").write_bytes(b"music")
    timeline = build_timeline(p)
    timeline["music"] = "score.wav"  # Native mode now mixes the score bed too.

    renderer = RecordingRenderer()
    renderer.render(
        p,
        timeline,
        str(p.path("output", "native.mp4")),
    )

    assert renderer.audio_sources == [b"native-one", b"native-two"]
    assert renderer.audio_fits == [
        ("sh-001.native.wav", 3.0),
        ("sh-002.native.wav", 2.0),
    ]
    assert renderer.mux_music is not None


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="ffmpeg and ffprobe are required for real assembly timing verification",
)
def test_native_renderer_fits_mismatched_tracks_to_shot_boundaries(tmp_path, monkeypatch):
    p = _project(tmp_path)
    p.model_config["audio_mode"] = "native_video"
    p.path("assets", "audio", "audio.json").write_text(json.dumps(NATIVE_AUDIO))
    for shot in SHOTS["shots"]:
        FakeImageGen().generate(
            "assembly timing keyframe",
            out_path=str(p.path("storyboard", "keyframes", shot["keyframe"])),
        )

    for filename, frequency, duration, sample_rate in (
        ("sh-001.native.wav", 440, 4.0, 16_000),
        ("sh-002.native.wav", 880, 1.0, 22_050),
    ):
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "lavfi", "-i", f"sine=frequency={frequency}:duration={duration}",
                "-ar", str(sample_rate), "-ac", "1", "-c:a", "pcm_s16le",
                str(p.path("assets", "audio", filename)),
            ],
            check=True,
        )

    monkeypatch.setattr(ffmpeg_edit, "WIDTH", 160)
    monkeypatch.setattr(ffmpeg_edit, "HEIGHT", 90)
    monkeypatch.setattr(ffmpeg_edit, "FPS", 5)
    output = p.path("output", "timing.mp4")

    FFmpegRenderer().render(p, build_timeline(p), str(output))

    probe = json.loads(subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration:stream=codec_type,sample_rate,channels",
            "-of", "json", str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout)
    assert float(probe["format"]["duration"]) == pytest.approx(5.0, abs=0.15)
    audio_streams = [
        stream for stream in probe["streams"] if stream["codec_type"] == "audio"
    ]
    assert len(audio_streams) == 1
    assert audio_streams[0]["sample_rate"] == "48000"
    assert audio_streams[0]["channels"] == 2

    def window_rms(start: float) -> float:
        raw = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-ss", str(start), "-t", "0.2",
                "-i", str(output), "-map", "0:a:0", "-ac", "1", "-ar", "8000",
                "-f", "s16le", "pipe:1",
            ],
            check=True,
            capture_output=True,
        ).stdout
        samples = struct.unpack(f"<{len(raw) // 2}h", raw)
        return (sum(sample * sample for sample in samples) / len(samples)) ** 0.5

    first_tone = window_rms(0.5)
    second_tone = window_rms(3.5)
    padded_gap = window_rms(4.5)
    assert first_tone > 300
    assert padded_gap < first_tone * 0.1
    assert second_tone > 300


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is required")
def test_audio_fitting_trims_long_input_and_normalizes_wav(tmp_path):
    source = tmp_path / "long-mono-16k.wav"
    output = tmp_path / "fitted-stereo-48k.wav"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(source),
        ],
        check=True,
    )

    FFmpegRenderer()._fit_audio_segment(source, 2.0, output)

    with wave.open(str(output), "rb") as fitted:
        assert fitted.getframerate() == 48_000
        assert fitted.getnchannels() == 2
        assert fitted.getnframes() / fitted.getframerate() == pytest.approx(2.0, abs=0.01)


def test_renderer_rejects_missing_native_track(tmp_path):
    p = _project(tmp_path)
    p.model_config["audio_mode"] = "native_video"
    p.path("assets", "audio", "audio.json").write_text(json.dumps(NATIVE_AUDIO))
    for shot in SHOTS["shots"]:
        p.path("storyboard", "keyframes", shot["keyframe"]).write_bytes(b"png")

    timeline = build_timeline(p)

    with pytest.raises(RuntimeError, match="native audio artifact missing for shot sh-001"):
        RecordingRenderer().render(p, timeline, str(p.path("output", "out.mp4")))


def test_native_renderer_rejects_audio_path_outside_project_audio_folder(tmp_path):
    p = _project(tmp_path)
    p.model_config["audio_mode"] = "native_video"
    native = json.loads(json.dumps(NATIVE_AUDIO))
    native["tracks"][0]["file"] = "../outside.wav"
    p.path("assets", "audio", "audio.json").write_text(json.dumps(native))
    p.path("assets", "outside.wav").write_bytes(b"must-not-be-read")
    for shot in SHOTS["shots"]:
        p.path("storyboard", "keyframes", shot["keyframe"]).write_bytes(b"png")

    with pytest.raises(RuntimeError, match="native audio artifact missing for shot sh-001"):
        RecordingRenderer().render(
            p,
            build_timeline(p),
            str(p.path("output", "out.mp4")),
        )
