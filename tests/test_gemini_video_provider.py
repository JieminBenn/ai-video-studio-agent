from pathlib import Path
from types import SimpleNamespace

from studio_agent.cli import build_providers
from studio_agent.providers.gemini_video import GeminiVeoVideoGen


def test_build_providers_selects_gemini_veo_lazily(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    providers = build_providers({
        "llm": "fake", "image": "fake", "video": "gemini-veo",
        "video_model": "veo-3.1-generate-preview", "vlm": "fake",
        "tts": "fake", "music": "fake",
    })

    assert isinstance(providers.video, GeminiVeoVideoGen)
    assert providers.video.capabilities.supports_reference_images is True
    assert providers.video.capabilities.supports_last_frame is True
    assert providers.video.capabilities.max_image_inputs == 4


def test_gemini_veo_reports_native_audio_only_when_enabled():
    assert GeminiVeoVideoGen(generate_audio=True).capabilities.supports_native_audio
    assert not GeminiVeoVideoGen(generate_audio=False).capabilities.supports_native_audio


def test_gemini_veo_passes_keyframe_references_last_frame_and_saves(tmp_path):
    paths = []
    for name in ("key.png", "ref.png", "last.png"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        paths.append(path)

    calls = []
    video = SimpleNamespace(save=lambda path: Path(path).write_bytes(b"veo-video"), uri="https://veo/video")
    operation = SimpleNamespace(done=True, response=SimpleNamespace(
        generated_videos=[SimpleNamespace(video=video)]
    ), name="operations/veo-1")

    def responder(**kwargs):
        calls.append(kwargs)
        return operation

    gen = GeminiVeoVideoGen(
        model="veo-3.1-generate-preview",
        responder=responder,
        image_loader=lambda path: f"image:{Path(path).name}",
        downloader=lambda generated, out_path: generated.save(out_path),
        poll_interval_s=0,
    )
    out = tmp_path / "clip.mp4"
    result = gen.generate(
        "camera pushes in",
        out_path=str(out),
        keyframe_path=str(paths[0]),
        reference_images=[str(paths[1])],
        last_frame_ref=str(paths[2]),
        duration_s=6,
        aspect_ratio="16:9",
    )

    assert calls[0]["image"] == "image:key.png"
    assert calls[0]["reference_images"] == ["image:ref.png"]
    assert calls[0]["last_frame"] == "image:last.png"
    assert calls[0]["duration_s"] == 6
    assert out.read_bytes() == b"veo-video"
    assert result.meta["operation_name"] == "operations/veo-1"
    assert result.meta["reference_images"] == [str(paths[1])]


def test_gemini_veo_none_duration_defers_to_model_and_bills_at_max(tmp_path):
    keyframe = tmp_path / "key.png"
    keyframe.write_bytes(b"kf")
    calls = []
    video = SimpleNamespace(save=lambda path: Path(path).write_bytes(b"v"), uri="https://veo/v")
    operation = SimpleNamespace(done=True, response=SimpleNamespace(
        generated_videos=[SimpleNamespace(video=video)]
    ), name="operations/veo-2")

    def responder(**kwargs):
        calls.append(kwargs)
        return operation

    gen = GeminiVeoVideoGen(
        model="veo-3.1-generate-preview",
        responder=responder,
        image_loader=lambda path: f"image:{Path(path).name}",
        downloader=lambda generated, out_path: generated.save(out_path),
        poll_interval_s=0,
    )
    out = tmp_path / "clip.mp4"
    result = gen.generate(
        "prompt", out_path=str(out), keyframe_path=str(keyframe), duration_s=None
    )
    # The duration reaches the operation as None (the real path omits the config field).
    assert calls[0]["duration_s"] is None
    assert result.meta["duration_s"] is None
    # Cost never undercounts: billed at the model's max duration.
    expected = round(gen.capabilities.max_duration_s * gen.cost_per_second, 4)
    assert result.cost_usd == expected


def test_gemini_veo_capabilities_declare_veo_duration_band():
    caps = GeminiVeoVideoGen().capabilities
    assert caps.min_duration_s == 4
    assert caps.max_duration_s == 8
