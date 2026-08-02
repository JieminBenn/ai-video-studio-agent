import io
from pathlib import Path

from studio_agent.cli import build_providers
from studio_agent.providers import xai_video
from studio_agent.providers.xai_video import XAIVideoGen, _download_url


def test_download_url_sends_browser_user_agent(tmp_path, monkeypatch):
    # imgen.x.ai (xAI's CDN that serves returned media URLs) rejects urllib's default
    # "Python-urllib/x.y" User-Agent with HTTP 403, so the download must send a real one.
    captured = {}

    class _Resp:
        def __enter__(self):
            return io.BytesIO(b"mp4-bytes")

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, *args, **kwargs):
        captured["user_agent"] = request.get_header("User-agent")
        return _Resp()

    monkeypatch.setattr(xai_video, "urlopen", fake_urlopen)
    out = tmp_path / "clip.mp4"
    _download_url("https://imgen.x.ai/out.mp4", str(out))

    assert captured["user_agent"], "download must set a User-Agent header"
    assert "python-urllib" not in captured["user_agent"].lower()
    assert out.read_bytes() == b"mp4-bytes"


def test_build_providers_selects_xai_video_lazily(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    providers = build_providers({
        "llm": "fake", "image": "fake", "video": "xai",
        "video_model": "grok-imagine-video", "vlm": "fake",
        "tts": "fake", "music": "fake",
    })

    assert isinstance(providers.video, XAIVideoGen)
    assert providers.video.capabilities.supports_reference_images is False
    assert providers.video.capabilities.supports_last_frame is False
    assert providers.video.capabilities.max_image_inputs == 1


def test_xai_video_creates_polls_and_downloads_image_to_video(tmp_path, monkeypatch):
    keyframe = tmp_path / "key.png"
    keyframe.write_bytes(b"keyframe")
    requests = []

    def request(method, url, *, api_key, json_body=None, timeout_s=None):
        requests.append((method, url, api_key, json_body))
        if method == "POST":
            return {"request_id": "video-1"}
        return {"status": "done", "video": {"url": "https://x.ai/clip.mp4"}}

    def download(url, out_path):
        assert url == "https://x.ai/clip.mp4"
        Path(out_path).write_bytes(b"xai-video")

    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    gen = XAIVideoGen(requester=request, downloader=download, poll_interval_s=0)
    out = tmp_path / "clip.mp4"
    result = gen.generate(
        "slow orbit",
        out_path=str(out),
        keyframe_path=str(keyframe),
        duration_s=7,
        aspect_ratio="16:9",
    )

    body = requests[0][3]
    assert requests[0][0] == "POST"
    assert body["model"] == "grok-imagine-video"
    assert body["duration"] == 7
    assert body["image"]["url"].startswith("data:image/png;base64,")
    assert requests[1][0] == "GET"
    assert out.read_bytes() == b"xai-video"
    assert result.meta["request_id"] == "video-1"
    assert result.meta["video_url"] == "https://x.ai/clip.mp4"
    assert result.cost_usd == 0.492  # 7 x $0.07 output seconds + $0.002 keyframe input.


def test_xai_video_15_preview_is_image_to_video_only():
    gen = XAIVideoGen(model="grok-imagine-video-1.5-preview")
    assert gen.capabilities.max_image_inputs == 1
