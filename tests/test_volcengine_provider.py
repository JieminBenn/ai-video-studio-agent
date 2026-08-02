"""Offline tests for the Volcengine Ark Seedance video provider.

Live Ark video generation is paid + non-deterministic, so these tests inject a fake
HTTP transport and downloader. No network or paid provider is used here.
"""

import base64
import io
import socket
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

from studio_agent.cli import build_providers
from studio_agent.providers.volcengine import (
    BYTEPLUS_BASE_URL,
    BYTEPLUS_DEFAULT_MODEL,
    DEFAULT_MODEL,
    ArkSeedanceVideoGen,
    _guard_inputs,
    _request_json,
)


def test_request_json_maps_read_timeout_to_actionable_runtimeerror(monkeypatch):
    """A read-phase socket timeout raises a raw TimeoutError that is *not* a
    URLError subclass, so it must be caught explicitly and surfaced with a clear,
    actionable message (this is the bare 'The read operation timed out' bug)."""

    def fake_urlopen(request, timeout=None):
        raise TimeoutError("The read operation timed out")

    monkeypatch.setattr("studio_agent.providers.volcengine.urlopen", fake_urlopen)

    with pytest.raises(RuntimeError) as excinfo:
        _request_json(
            "POST",
            "https://ark.example/api/v3/images/generations",
            api_key="test-key",
            json_body={"prompt": "x"},
            timeout_s=5.0,
        )

    message = str(excinfo.value).lower()
    assert "timed out" in message or "timeout" in message
    # actionable: points at the timeout knob so the user can raise it
    assert "timeout_s" in message or "image_timeout_s" in message or "video_timeout_s" in message


class _FakeJsonResponse:
    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


def _dns_error() -> URLError:
    # The exact failure seen in the field: getaddrinfo could not resolve the host.
    return URLError(socket.gaierror(8, "nodename nor servname provided, or not known"))


def test_request_json_retries_transient_dns_failure_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(request, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise _dns_error()
        return _FakeJsonResponse(b'{"ok": true}')

    monkeypatch.setattr("studio_agent.providers.volcengine.urlopen", fake_urlopen)
    monkeypatch.setattr("studio_agent.providers.volcengine.time.sleep", lambda *_: None)

    result = _request_json("GET", "https://ark.example/api/v3/x", api_key="k")

    assert result == {"ok": True}
    assert calls["n"] == 3  # rode through two transient blips


def test_request_json_gives_up_after_retries_on_persistent_dns_failure(monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(request, timeout=None):
        calls["n"] += 1
        raise _dns_error()

    monkeypatch.setattr("studio_agent.providers.volcengine.urlopen", fake_urlopen)
    monkeypatch.setattr("studio_agent.providers.volcengine.time.sleep", lambda *_: None)

    with pytest.raises(RuntimeError) as excinfo:
        _request_json("GET", "https://ark.example/api/v3/x", api_key="k")

    assert "Ark API request failed" in str(excinfo.value)
    assert calls["n"] == 3  # 1 initial + 2 retries, then surfaces the error


def test_request_json_does_not_retry_client_error(monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(request, timeout=None):
        calls["n"] += 1
        raise HTTPError("x", 401, "unauthorized", {}, io.BytesIO(b"bad key"))

    monkeypatch.setattr("studio_agent.providers.volcengine.urlopen", fake_urlopen)
    monkeypatch.setattr("studio_agent.providers.volcengine.time.sleep", lambda *_: None)

    with pytest.raises(RuntimeError) as excinfo:
        _request_json("GET", "https://ark.example/api/v3/x", api_key="k")

    assert "401" in str(excinfo.value)
    assert calls["n"] == 1  # auth errors are terminal — never retried


def test_build_providers_selects_volcengine_video():
    providers = build_providers({
        "llm": "fake",
        "image": "fake",
        "video": "volcengine",
        "video_model": "doubao-seedance-2-0-fast-260128",
        "tts": "fake",
        "music": "fake",
        "vlm": "fake",
    })

    assert isinstance(providers.video, ArkSeedanceVideoGen)
    assert providers.video.model == "doubao-seedance-2-0-fast-260128"


def test_build_providers_selects_byteplus_video():
    providers = build_providers({
        "llm": "fake",
        "image": "fake",
        "video": "byteplus",
        "video_model": "dreamina-seedance-2-0-fast-260128",
        "video_base_url": BYTEPLUS_BASE_URL,
        "video_api_key_env": "BYTEPLUS_ARK_API_KEY",
        "video_cost_per_1k_tokens_usd": 0.0056,
        "video_timeout_s": 240.0,
        "video_include_reference_images": False,
        "video_include_last_frame_ref": False,
        "tts": "fake",
        "music": "fake",
        "vlm": "fake",
    })

    assert isinstance(providers.video, ArkSeedanceVideoGen)
    assert providers.video.name == "byteplus"
    assert providers.video.model == "dreamina-seedance-2-0-fast-260128"
    assert providers.video.base_url == BYTEPLUS_BASE_URL
    assert providers.video.api_key_env == "BYTEPLUS_ARK_API_KEY"
    assert providers.video.cost_per_1k_tokens_usd == 0.0056
    assert providers.video.cost_per_million_tokens_cny is None
    assert providers.video.timeout_s == 240.0
    assert providers.video.include_reference_images is False
    assert providers.video.include_last_frame_ref is False


def test_volcengine_constructs_without_network_or_key():
    gen = ArkSeedanceVideoGen()
    assert gen.name == "volcengine"
    assert gen.model == DEFAULT_MODEL


def test_ark_native_audio_capability_is_model_aware():
    assert ArkSeedanceVideoGen(
        model="doubao-seedance-2-0-fast-260128", generate_audio=True
    ).capabilities.supports_native_audio
    assert ArkSeedanceVideoGen(
        model="doubao-seedance-1-5-pro-251215", generate_audio=True
    ).capabilities.supports_native_audio
    assert not ArkSeedanceVideoGen(
        model="doubao-seedance-1-0-pro-250528", generate_audio=True
    ).capabilities.supports_native_audio


def test_ark_capabilities_do_not_advertise_inputs_the_adapter_drops():
    capabilities = ArkSeedanceVideoGen(
        include_reference_images=True,
        include_last_frame_ref=True,
    ).capabilities

    assert capabilities.supports_reference_images is False
    assert capabilities.supports_last_frame is False


def test_volcengine_creates_polls_and_downloads_output(tmp_path, monkeypatch):
    keyframe = tmp_path / "keyframe.png"
    reference = tmp_path / "reference.png"
    previous = tmp_path / "previous.png"
    keyframe.write_bytes(b"keyframe")
    reference.write_bytes(b"reference")
    previous.write_bytes(b"previous")
    out = tmp_path / "clip.mp4"

    requests = []
    poll_count = 0

    def request(method, url, *, api_key, json_body=None, timeout_s=None):
        nonlocal poll_count
        requests.append({
            "method": method,
            "url": url,
            "api_key": api_key,
            "json_body": json_body,
        })
        if method == "POST":
            return {"id": "cgt-test-task"}
        poll_count += 1
        if poll_count == 1:
            return {"id": "cgt-test-task", "status": "running"}
        return {
            "id": "cgt-test-task",
            "status": "succeeded",
            "model": "doubao-seedance-2-0-fast-260128",
            "content": {"video_url": "https://video.example/out.mp4"},
            "usage": {"completion_tokens": 123456, "total_tokens": 123456},
            "resolution": "720p",
            "ratio": "16:9",
            "duration": 4,
        }

    downloads = []

    def download(url, out_path):
        downloads.append((url, out_path))
        Path(out_path).write_bytes(b"downloaded-video")

    monkeypatch.setenv("ARK_API_KEY", "test-key")
    gen = ArkSeedanceVideoGen(
        model="doubao-seedance-2-0-fast-260128",
        requester=request,
        downloader=download,
        poll_interval_s=0,
    )

    result = gen.generate(
        "wide shot of Mara",
        out_path=str(out),
        keyframe_path=str(keyframe),
        reference_images=[str(reference)],
        last_frame_ref=str(previous),
        duration_s=2.0,
        seed=7,
    )

    assert [r["method"] for r in requests] == ["POST", "GET", "GET"]
    create_body = requests[0]["json_body"]
    assert create_body["model"] == "doubao-seedance-2-0-fast-260128"
    assert create_body["duration"] == 4  # Seedance 2.0 minimum is 4 seconds.
    assert create_body["resolution"] == "720p"
    assert create_body["ratio"] == "16:9"
    assert create_body["generate_audio"] is False
    assert create_body["return_last_frame"] is True
    assert "seed" not in create_body  # Seedance 2.0 does not support seed.

    content = create_body["content"]
    assert content[0] == {"type": "text", "text": "wide shot of Mara"}
    # Ark forbids mixing first/last-frame content with reference media, so the keyframe
    # is sent alone as the first frame and the reference media is dropped.
    assert [item["role"] for item in content[1:]] == ["first_frame"]
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert (
        content[1]["image_url"]["url"].split(",", 1)[1]
        == base64.b64encode(b"keyframe").decode("ascii")
    )

    assert requests[0]["api_key"] == "test-key"
    assert downloads == [("https://video.example/out.mp4", str(out))]
    assert out.read_bytes() == b"downloaded-video"
    assert result.content == str(out)
    assert result.provider == "volcengine"
    assert result.meta["task_id"] == "cgt-test-task"
    assert result.meta["video_url"] == "https://video.example/out.mp4"
    assert result.meta["usage"]["completion_tokens"] == 123456
    assert result.meta["reference_images"] == []
    assert result.meta["last_frame_ref"] is None
    assert result.meta["dropped_reference_images"] == [str(reference)]
    assert result.meta["dropped_last_frame_ref"] == str(previous)


def test_volcengine_never_mixes_first_frame_with_reference_media(tmp_path, monkeypatch):
    """Regression: Ark rejects a request that mixes a first frame with reference media.

    HTTP 400 "first/last frame content cannot be mixed with reference media content".
    The provider must send the keyframe alone as the first frame.
    """
    keyframe = tmp_path / "keyframe.png"
    reference = tmp_path / "reference.png"
    previous = tmp_path / "previous.png"
    for path in (keyframe, reference, previous):
        path.write_bytes(path.name.encode("ascii"))
    out = tmp_path / "clip.mp4"

    bodies = []

    def request(method, url, *, api_key, json_body=None, timeout_s=None):
        if method == "POST":
            bodies.append(json_body)
            return {"id": "cgt-task"}
        return {
            "id": "cgt-task",
            "status": "succeeded",
            "content": {"video_url": "https://video.example/out.mp4"},
        }

    monkeypatch.setenv("ARK_API_KEY", "test-key")
    gen = ArkSeedanceVideoGen(
        requester=request,
        downloader=lambda url, out_path: Path(out_path).write_bytes(b"v"),
        poll_interval_s=0,
        include_reference_images=True,
        include_last_frame_ref=True,
    )

    gen.generate(
        "wide shot",
        out_path=str(out),
        keyframe_path=str(keyframe),
        reference_images=[str(reference)],
        last_frame_ref=str(previous),
    )

    roles = [item["role"] for item in bodies[0]["content"] if item["type"] == "image_url"]
    assert roles == ["first_frame"]
    assert "reference_image" not in roles


def test_volcengine_maps_privacy_moderation_error_to_actionable_message(tmp_path, monkeypatch):
    keyframe = tmp_path / "keyframe.png"
    keyframe.write_bytes(b"keyframe")
    out = tmp_path / "clip.mp4"

    moderation_detail = (
        '{"error":{"code":"InputImageSensitiveContentDetected.PrivacyInformation",'
        '"message":"The request failed because the input image may contain real person. '
        'Request id: 0217817","param":"","type":"BadRequest"}}'
    )

    def request(method, url, *, api_key, json_body=None, timeout_s=None):
        raise RuntimeError(f"Ark API HTTP 400: {moderation_detail}")

    monkeypatch.setenv("ARK_API_KEY", "test-key")
    gen = ArkSeedanceVideoGen(
        requester=request,
        downloader=lambda url, out_path: None,
        poll_interval_s=0,
    )

    with pytest.raises(RuntimeError) as excinfo:
        gen.generate(
            "wide shot",
            out_path=str(out),
            keyframe_path=str(keyframe),
            duration_s=4.0,
        )

    message = str(excinfo.value)
    # The raw moderation code is preserved for debugging,
    assert "PrivacyInformation" in message
    # but the surfaced error is actionable: it names the rejected keyframe and a remedy.
    lowered = message.lower()
    assert "real person" in lowered
    assert "keyframe" in lowered
    assert "styliz" in lowered or "non-photoreal" in lowered or "regenerate" in lowered


def test_byteplus_uses_international_endpoint_key_and_usd_cost(tmp_path, monkeypatch):
    keyframe = tmp_path / "keyframe.png"
    reference = tmp_path / "reference.png"
    previous = tmp_path / "previous.png"
    keyframe.write_bytes(b"keyframe")
    reference.write_bytes(b"reference")
    previous.write_bytes(b"previous")
    out = tmp_path / "clip.mp4"

    requests = []

    def request(method, url, *, api_key, json_body=None, timeout_s=None):
        requests.append({
            "method": method,
            "url": url,
            "api_key": api_key,
            "json_body": json_body,
        })
        if method == "POST":
            return {"id": "byteplus-task"}
        return {
            "id": "byteplus-task",
            "status": "succeeded",
            "model": "dreamina-seedance-2-0-fast-260128",
            "content": {"video_url": "https://video.example/byteplus.mp4"},
            "usage": {"completion_tokens": 2000},
        }

    def download(url, out_path):
        Path(out_path).write_bytes(b"byteplus-video")

    monkeypatch.setenv("BYTEPLUS_ARK_API_KEY", "byteplus-test-key")
    gen = ArkSeedanceVideoGen(
        model=BYTEPLUS_DEFAULT_MODEL,
        provider_name="byteplus",
        provider_label="BytePlus ModelArk",
        base_url=BYTEPLUS_BASE_URL,
        api_key_env="BYTEPLUS_ARK_API_KEY",
        requester=request,
        downloader=download,
        poll_interval_s=0,
        cost_per_million_tokens_cny=None,
        cost_per_1k_tokens_usd=0.0056,
        include_reference_images=False,
        include_last_frame_ref=False,
    )

    result = gen.generate(
        "wide shot of a lantern robot",
        out_path=str(out),
        keyframe_path=str(keyframe),
        reference_images=[str(reference)],
        last_frame_ref=str(previous),
    )

    assert requests[0]["url"] == f"{BYTEPLUS_BASE_URL}/contents/generations/tasks"
    assert requests[1]["url"] == f"{BYTEPLUS_BASE_URL}/contents/generations/tasks/byteplus-task"
    assert requests[0]["api_key"] == "byteplus-test-key"
    assert requests[0]["json_body"]["model"] == "dreamina-seedance-2-0-fast-260128"
    content = requests[0]["json_body"]["content"]
    assert [item["role"] for item in content[1:]] == ["first_frame"]
    assert result.provider == "byteplus"
    assert result.cost_usd == 0.0112
    assert result.meta["cost_per_1k_tokens_usd"] == 0.0056
    assert result.meta["estimated_cost_cny"] == 0.0
    assert result.meta["reference_images"] == []
    assert result.meta["last_frame_ref"] is None
    assert result.meta["dropped_reference_images"] == [str(reference)]
    assert result.meta["dropped_last_frame_ref"] == str(previous)
    assert out.read_bytes() == b"byteplus-video"


def test_volcengine_uses_reported_usage_for_cost(tmp_path, monkeypatch):
    keyframe = tmp_path / "keyframe.png"
    keyframe.write_bytes(b"keyframe")

    def request(method, url, *, api_key, json_body=None, timeout_s=None):
        if method == "POST":
            return {"id": "cgt-cost"}
        return {
            "id": "cgt-cost",
            "status": "succeeded",
            "content": {"video_url": "https://video.example/out.mp4"},
            "usage": {"completion_tokens": 1_000_000},
        }

    def download(url, out_path):
        Path(out_path).write_bytes(b"video")

    monkeypatch.setenv("ARK_API_KEY", "test-key")
    gen = ArkSeedanceVideoGen(
        requester=request,
        downloader=download,
        poll_interval_s=0,
        cost_per_million_tokens_cny=37.0,
        cny_to_usd=0.14,
    )

    result = gen.generate(
        "p",
        out_path=str(tmp_path / "out.mp4"),
        keyframe_path=str(keyframe),
    )

    assert result.cost_usd == 5.18
    assert result.meta["estimated_cost_cny"] == 37.0
    assert result.meta["cost_tracking"]["native_cost"] == 37.0
    assert result.meta["cost_tracking"]["native_currency"] == "CNY"
    assert result.meta["cost_tracking"]["usd_conversion_rate"] == 0.14
    assert result.meta["cost_tracking"]["usage"]["completion_tokens"] == 1_000_000


def test_volcengine_raises_on_failed_task(tmp_path, monkeypatch):
    keyframe = tmp_path / "keyframe.png"
    keyframe.write_bytes(b"keyframe")

    def request(method, url, *, api_key, json_body=None, timeout_s=None):
        if method == "POST":
            return {"id": "cgt-failed"}
        return {
            "id": "cgt-failed",
            "status": "failed",
            "error": {"code": "BadPrompt", "message": "prompt rejected"},
        }

    monkeypatch.setenv("ARK_API_KEY", "test-key")
    gen = ArkSeedanceVideoGen(requester=request, poll_interval_s=0)

    with pytest.raises(RuntimeError) as exc:
        gen.generate(
            "p",
            out_path=str(tmp_path / "out.mp4"),
            keyframe_path=str(keyframe),
        )

    assert "failed" in str(exc.value)
    assert "prompt rejected" in str(exc.value)


def test_volcengine_guards_too_many_images_offline():
    # The keyframe is the only image sent (Ark forbids mixing reference media with a
    # first frame), but the input guard still rejects over-limit batches defensively.
    paths = [f"img-{i}.png" for i in range(10)]

    with pytest.raises(ValueError) as exc:
        _guard_inputs(paths, 9, model=DEFAULT_MODEL, provider_label="volcengine")

    msg = str(exc.value)
    assert "at most 9" in msg
    assert "10" in msg


def test_volcengine_requires_keyframe(tmp_path):
    gen = ArkSeedanceVideoGen()

    with pytest.raises(ValueError) as exc:
        gen.generate("p", out_path=str(tmp_path / "out.mp4"))

    assert "keyframe_path" in str(exc.value)


def test_volcengine_rejects_bad_create_response(tmp_path, monkeypatch):
    keyframe = tmp_path / "keyframe.png"
    keyframe.write_bytes(b"keyframe")

    def request(method, url, *, api_key, json_body=None, timeout_s=None):
        return {"unexpected": "shape"}

    monkeypatch.setenv("ARK_API_KEY", "test-key")
    gen = ArkSeedanceVideoGen(requester=request, poll_interval_s=0)

    with pytest.raises(RuntimeError) as exc:
        gen.generate(
            "p",
            out_path=str(tmp_path / "out.mp4"),
            keyframe_path=str(keyframe),
        )

    assert "task id" in str(exc.value)


def test_seedance_duration_none_maps_to_model_default_sentinel():
    from studio_agent.providers.volcengine import _seedance_duration

    assert _seedance_duration(None) == -1
    assert _seedance_duration(-1) == -1
    assert _seedance_duration(2) == 4
    assert _seedance_duration(99) == 15
    assert _seedance_duration(8) == 8


def test_ark_capabilities_declare_seedance_duration_band():
    from studio_agent.providers.volcengine import ArkSeedanceVideoGen

    gen = ArkSeedanceVideoGen()
    assert gen.capabilities.min_duration_s == 4
    assert gen.capabilities.max_duration_s == 15
