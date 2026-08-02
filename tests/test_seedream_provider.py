from pathlib import Path

import pytest

from studio_agent.cli import build_providers
from studio_agent.providers.seedream import ArkSeedreamImageGen

PNG_BYTES = b"\x89PNG\r\n\x1a\nseedream-ref"


def test_build_providers_selects_doubao_seedream_image():
    providers = build_providers({
        "llm": "fake",
        "image": "doubao",
        "image_model": "doubao-seedream-4-0-250828",
        "video": "fake",
        "tts": "fake",
        "music": "fake",
        "vlm": "fake",
    })

    assert isinstance(providers.image, ArkSeedreamImageGen)
    assert providers.image.name == "doubao"
    assert providers.image.model == "doubao-seedream-4-0-250828"


def test_doubao_default_timeout_matches_other_image_providers():
    """Synchronous Seedream renders (heavy 2K turnaround/expression sheets) can
    take well over 60s; the default read timeout must match gemini/openai (120s)
    so a slow-but-valid render doesn't fail an already-paid bible stage."""
    providers = build_providers({
        "llm": "fake",
        "image": "doubao",
        "image_model": "doubao-seedream-4-0-250828",
        "video": "fake",
        "tts": "fake",
        "music": "fake",
        "vlm": "fake",
    })

    assert providers.image.timeout_s == 120.0


def test_build_providers_selects_byteplus_seedream_image():
    providers = build_providers({
        "llm": "fake",
        "image": "byteplus-seedream",
        "image_model": "seedream-4-0-250828",
        "image_base_url": "https://ark.ap-southeast.bytepluses.com/api/v3",
        "image_api_key_env": "BYTEPLUS_ARK_API_KEY",
        "video": "fake",
        "tts": "fake",
        "music": "fake",
        "vlm": "fake",
    })

    assert isinstance(providers.image, ArkSeedreamImageGen)
    assert providers.image.name == "byteplus-seedream"
    assert providers.image.api_key_env == "BYTEPLUS_ARK_API_KEY"
    assert providers.image.model == "seedream-4-0-250828"


def test_seedream_generates_and_downloads_image(tmp_path, monkeypatch):
    ref = tmp_path / "ref.png"
    ref.write_bytes(PNG_BYTES)
    out = tmp_path / "out.png"
    requests = []

    def request(url, *, api_key, json_body, timeout_s=None):
        requests.append({
            "url": url,
            "api_key": api_key,
            "json_body": json_body,
        })
        return {"data": [{"url": "https://image.example/out.png", "size": "2K"}], "usage": {}}

    downloads = []

    def download(url, out_path):
        downloads.append((url, out_path))
        Path(out_path).write_bytes(b"downloaded-image")

    monkeypatch.setenv("ARK_API_KEY", "test-key")
    gen = ArkSeedreamImageGen(requester=request, downloader=download, poll_interval_s=0)

    result = gen.generate(
        "reference-conditioned keyframe",
        out_path=str(out),
        reference_images=[str(ref)],
        seed=7,
    )

    assert [r["url"] for r in requests] == [
        "https://ark.cn-beijing.volces.com/api/v3/images/generations"
    ]
    body = requests[0]["json_body"]
    assert body["model"] == "doubao-seedream-4-0-250828"
    assert body["prompt"] == "reference-conditioned keyframe"
    assert body["size"] == "2K"
    assert body["response_format"] == "url"
    assert "output_format" not in body
    assert body["image"].startswith("data:image/png;base64,")
    assert body["watermark"] is False
    assert body["seed"] == 7
    assert requests[0]["api_key"] == "test-key"
    assert downloads == [("https://image.example/out.png", str(out))]
    assert out.read_bytes() == b"downloaded-image"
    assert result.provider == "doubao"
    assert result.meta["reference_images"] == [str(ref)]
    assert result.meta["endpoint"] == "images/generations"


def test_seedream_omits_output_format_by_default(tmp_path, monkeypatch):
    """Seedream 4.0 rejects `output_format`; it must not be sent unless configured."""
    out = tmp_path / "out.png"
    requests = []

    def request(url, *, api_key, json_body, timeout_s=None):
        requests.append(json_body)
        return {"data": [{"b64_json": "aGk="}]}

    monkeypatch.setenv("ARK_API_KEY", "test-key")
    gen = ArkSeedreamImageGen(requester=request, downloader=lambda *a: None)

    gen.generate("p", out_path=str(out))

    assert "output_format" not in requests[0]


def test_seedream_sends_output_format_when_configured(tmp_path, monkeypatch):
    out = tmp_path / "out.png"
    requests = []

    def request(url, *, api_key, json_body, timeout_s=None):
        requests.append(json_body)
        return {"data": [{"b64_json": "aGk="}]}

    monkeypatch.setenv("ARK_API_KEY", "test-key")
    gen = ArkSeedreamImageGen(
        output_format="png", requester=request, downloader=lambda *a: None
    )

    gen.generate("p", out_path=str(out))

    assert requests[0]["output_format"] == "png"


def test_build_providers_passes_image_output_format():
    providers = build_providers({
        "llm": "fake",
        "image": "doubao",
        "image_output_format": "jpeg",
        "video": "fake",
        "tts": "fake",
        "music": "fake",
        "vlm": "fake",
    })

    assert isinstance(providers.image, ArkSeedreamImageGen)
    assert providers.image.output_format == "jpeg"


def test_seedream_normalizes_short_model_aliases(tmp_path, monkeypatch):
    out = tmp_path / "out.png"
    requests = []

    def request(url, *, api_key, json_body, timeout_s=None):
        requests.append(json_body)
        return {"data": [{"url": "https://image.example/out.png"}]}

    def download(url, out_path):
        Path(out_path).write_bytes(b"downloaded-image")

    monkeypatch.setenv("ARK_API_KEY", "test-key")
    gen = ArkSeedreamImageGen(
        model="doubao-seedream-4-0",
        requester=request,
        downloader=download,
        poll_interval_s=0,
    )

    result = gen.generate("p", out_path=str(out))

    assert gen.model == "doubao-seedream-4-0-250828"
    assert requests[0]["model"] == "doubao-seedream-4-0-250828"
    assert result.model == "doubao-seedream-4-0-250828"


def test_seedream_requires_key(tmp_path, monkeypatch):
    monkeypatch.delenv("ARK_API_KEY", raising=False)
    gen = ArkSeedreamImageGen()

    with pytest.raises(RuntimeError, match="ARK_API_KEY"):
        gen.generate("p", out_path=str(tmp_path / "out.png"))
