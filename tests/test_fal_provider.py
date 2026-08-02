"""Offline tests for the fal reference-to-video provider.

The live fal call is paid + non-deterministic, so tests monkeypatch the fal client and
the video download step. No network or paid provider is used here.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from studio_agent.cli import build_providers
from studio_agent.providers.fal import DEFAULT_MODEL, FalReferenceVideoGen


def test_build_providers_selects_fal_video():
    providers = build_providers({
        "llm": "fake",
        "image": "fake",
        "video": "fal",
        "video_model": "bytedance/seedance-2.0/fast/reference-to-video",
        "tts": "fake",
        "music": "fake",
        "vlm": "fake",
    })

    assert isinstance(providers.video, FalReferenceVideoGen)
    assert providers.video.model == "bytedance/seedance-2.0/fast/reference-to-video"


def test_fal_constructs_without_network_or_key():
    gen = FalReferenceVideoGen()
    assert gen.name == "fal"
    assert gen.model == DEFAULT_MODEL


def test_fal_reports_native_audio_only_when_enabled():
    assert FalReferenceVideoGen(generate_audio=True).capabilities.supports_native_audio
    assert not FalReferenceVideoGen(generate_audio=False).capabilities.supports_native_audio


def test_fal_uploads_inputs_subscribes_and_downloads_output(tmp_path, monkeypatch):
    keyframe = tmp_path / "keyframe.png"
    reference = tmp_path / "reference.png"
    previous = tmp_path / "previous.png"
    keyframe.write_bytes(b"keyframe")
    reference.write_bytes(b"reference")
    previous.write_bytes(b"previous")
    out = tmp_path / "clip.mp4"

    uploads = []
    subscribes = []

    def upload_file(path):
        uploads.append(path)
        return f"https://cdn.example/{Path(path).name}"

    def subscribe(model, *, arguments, with_logs=False, on_queue_update=None):
        subscribes.append({
            "model": model,
            "arguments": arguments,
            "with_logs": with_logs,
            "on_queue_update": on_queue_update,
        })
        return {"video": {"url": "https://video.example/out.mp4"}, "seed": 123}

    monkeypatch.setitem(
        sys.modules,
        "fal_client",
        SimpleNamespace(upload_file=upload_file, subscribe=subscribe),
    )
    monkeypatch.setenv("FAL_KEY", "test-key")

    downloads = []

    def download(url, out_path):
        downloads.append((url, out_path))
        Path(out_path).write_bytes(b"downloaded-video")

    gen = FalReferenceVideoGen(model="model-id", downloader=download)

    result = gen.generate(
        "wide shot of Mara",
        out_path=str(out),
        keyframe_path=str(keyframe),
        reference_images=[str(reference)],
        last_frame_ref=str(previous),
        seed=7,
        duration_s=2.0,
        aspect_ratio="16:9",
    )

    assert uploads == [str(keyframe), str(reference), str(previous)]
    args = subscribes[0]["arguments"]
    assert subscribes[0]["model"] == "model-id"
    assert args["image_urls"] == [
        "https://cdn.example/keyframe.png",
        "https://cdn.example/reference.png",
        "https://cdn.example/previous.png",
    ]
    assert args["duration"] == "4"  # fal minimum is 4 seconds.
    assert args["aspect_ratio"] == "16:9"
    assert args["resolution"] == "720p"
    assert args["generate_audio"] is False
    assert args["seed"] == 7
    assert "@Image1" in args["prompt"] and "@Image2" in args["prompt"]
    assert downloads == [("https://video.example/out.mp4", str(out))]
    assert out.read_bytes() == b"downloaded-video"
    assert result.meta["reference_images"] == [str(reference)]
    assert result.meta["last_frame_ref"] == str(previous)
    assert result.meta["video_url"] == "https://video.example/out.mp4"


def test_fal_labels_target_state_reference_images(tmp_path, monkeypatch):
    keyframe = tmp_path / "keyframe.png"
    target = tmp_path / "dragon.png"
    keyframe.write_bytes(b"keyframe")
    target.write_bytes(b"dragon")
    out = tmp_path / "clip.mp4"

    def upload_file(path):
        return f"https://cdn.example/{Path(path).name}"

    subscribed = []

    def subscribe(model, *, arguments):
        subscribed.append(arguments)
        return {"video": {"url": "https://video.example/out.mp4"}, "seed": 123}

    monkeypatch.setitem(
        sys.modules,
        "fal_client",
        SimpleNamespace(upload_file=upload_file, subscribe=subscribe),
    )
    monkeypatch.setenv("FAL_KEY", "test-key")

    gen = FalReferenceVideoGen(
        model="model-id",
        downloader=lambda _url, path: Path(path).write_bytes(b"downloaded-video"),
    )

    result = gen.generate(
        "A man transforms into the approved dragon.",
        out_path=str(out),
        keyframe_path=str(keyframe),
        target_state_reference_images=[str(target)],
        duration_s=4.0,
    )

    prompt = subscribed[0]["prompt"]
    assert "@Image2" in prompt
    assert "target-state" in prompt.lower()
    assert subscribed[0]["image_urls"] == [
        "https://cdn.example/keyframe.png",
        "https://cdn.example/dragon.png",
    ]
    assert result.meta["target_state_reference_images"] == [str(target)]


def test_fal_generate_audio_call_override_disables_enabled_constructor(tmp_path):
    keyframe = tmp_path / "keyframe.png"
    keyframe.write_bytes(b"keyframe")
    out = tmp_path / "clip.mp4"
    subscribed = []

    def subscribe(model, *, arguments):
        subscribed.append(arguments)
        return {"video": {"url": "https://video.example/out.mp4"}, "seed": 123}

    gen = FalReferenceVideoGen(
        generate_audio=True,
        downloader=lambda _url, path: Path(path).write_bytes(b"downloaded-video"),
    )
    gen._client = SimpleNamespace(
        upload_file=lambda path: f"https://cdn.example/{Path(path).name}",
        subscribe=subscribe,
    )

    result = gen.generate(
        "wide shot of Mara",
        out_path=str(out),
        keyframe_path=str(keyframe),
        generate_audio=False,
    )

    assert subscribed[0]["generate_audio"] is False
    assert result.meta["generate_audio"] is False


def test_fal_guards_too_many_images_offline(tmp_path):
    keyframe = tmp_path / "keyframe.png"
    keyframe.write_bytes(b"keyframe")
    refs = []
    for i in range(9):
        ref = tmp_path / f"ref-{i}.png"
        ref.write_bytes(b"ref")
        refs.append(str(ref))

    gen = FalReferenceVideoGen(max_image_inputs=9)

    with pytest.raises(ValueError) as exc:
        gen.generate("p", out_path=str(tmp_path / "out.mp4"),
                     keyframe_path=str(keyframe), reference_images=refs)

    msg = str(exc.value)
    assert "at most 9" in msg
    assert "10" in msg


def test_fal_duration_none_means_auto_and_cost_still_computes(tmp_path, monkeypatch):
    from studio_agent.providers.fal import _billed_seconds, _fal_duration

    assert _fal_duration(None) == "auto"
    assert _fal_duration(2.0) == "4"
    assert _fal_duration(99) == "15"
    # "auto" duration: cost uses the service-reported seconds, else the band midpoint.
    assert _billed_seconds("auto", {"duration": 8}) == 8.0
    assert _billed_seconds("auto", {}) == (4 + 15) / 2
    assert _billed_seconds("6", {}) == 6.0

    keyframe = tmp_path / "kf.png"
    keyframe.write_bytes(b"kf")
    out = tmp_path / "clip.mp4"
    subscribes = []

    def subscribe(model, *, arguments, with_logs=False, on_queue_update=None):
        subscribes.append(arguments)
        return {"video": {"url": "https://video.example/out.mp4"}}

    monkeypatch.setitem(
        sys.modules,
        "fal_client",
        SimpleNamespace(upload_file=lambda p: f"https://cdn.example/{Path(p).name}", subscribe=subscribe),
    )
    monkeypatch.setenv("FAL_KEY", "test-key")
    gen = FalReferenceVideoGen(
        model="model-id",
        downloader=lambda url, out_path: Path(out_path).write_bytes(b"v"),
    )
    result = gen.generate(
        "prompt", out_path=str(out), keyframe_path=str(keyframe), duration_s=None
    )
    assert subscribes[0]["duration"] == "auto"
    assert result.cost_usd > 0  # midpoint estimate, no crash on the non-numeric sentinel


def test_fal_capabilities_declare_seedance_duration_band():
    gen = FalReferenceVideoGen(model="model-id")
    assert gen.capabilities.min_duration_s == 4
    assert gen.capabilities.max_duration_s == 15
