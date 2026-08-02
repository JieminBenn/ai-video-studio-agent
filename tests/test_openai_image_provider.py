import base64

import pytest

from studio_agent.cli import build_providers
from studio_agent.providers.openai_image import OpenAIImageGen


PNG_BYTES = b"\x89PNG\r\n\x1a\nopenai-image"


def test_build_providers_selects_openai_image():
    providers = build_providers({
        "llm": "fake",
        "image": "openai",
        "image_model": "gpt-image-2",
        "video": "fake",
        "tts": "fake",
        "music": "fake",
        "vlm": "fake",
    })

    assert isinstance(providers.image, OpenAIImageGen)
    assert providers.image.model == "gpt-image-2"


def test_openai_capabilities_report_gpt_image_prompt_limit():
    capabilities = OpenAIImageGen().capabilities

    assert capabilities.max_prompt_length == 32000
    assert capabilities.prompt_length_unit == "characters"


@pytest.mark.parametrize(
    ("model", "expected_limit"),
    [
        ("dall-e-2", 1000),
        ("dall-e-3", 4000),
        ("future-unverified-image-model", None),
    ],
)
def test_openai_prompt_limit_is_model_specific(model, expected_limit):
    assert OpenAIImageGen(model=model).capabilities.max_prompt_length == expected_limit


def test_openai_image_generations_request_writes_b64_output(tmp_path, monkeypatch):
    requests = []

    def request(url, *, api_key, json_body, timeout_s):
        requests.append({"url": url, "api_key": api_key, "json_body": json_body})
        return {"data": [{"b64_json": base64.b64encode(PNG_BYTES).decode("ascii")}]}

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    out = tmp_path / "image.png"
    gen = OpenAIImageGen(json_requester=request)

    result = gen.generate("a cinematic keyframe", out_path=str(out), seed=42)

    assert requests[0]["url"].endswith("/images/generations")
    assert requests[0]["api_key"] == "test-key"
    assert requests[0]["json_body"]["model"] == "gpt-image-2"
    assert requests[0]["json_body"]["prompt"] == "a cinematic keyframe"
    assert requests[0]["json_body"]["size"] == "1536x1024"
    assert out.read_bytes() == PNG_BYTES
    assert result.provider == "openai"
    assert result.meta["endpoint"] == "generations"
    assert result.meta["seed"] == 42


def test_openai_image_edits_send_reference_files(tmp_path, monkeypatch):
    ref = tmp_path / "ref.png"
    ref.write_bytes(PNG_BYTES)
    requests = []

    def multipart(url, *, api_key, fields, files, timeout_s):
        requests.append({
            "url": url,
            "api_key": api_key,
            "fields": fields,
            "files": files,
        })
        return {"data": [{"b64_json": base64.b64encode(PNG_BYTES).decode("ascii")}]}

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    out = tmp_path / "edited.png"
    gen = OpenAIImageGen(multipart_requester=multipart)

    result = gen.generate(
        "match the uploaded face",
        out_path=str(out),
        reference_images=[str(ref)],
    )

    assert requests[0]["url"].endswith("/images/edits")
    assert requests[0]["fields"]["model"] == "gpt-image-2"
    assert requests[0]["files"][0][0] == "image[]"
    assert requests[0]["files"][0][1] == "ref.png"
    assert requests[0]["files"][0][2] == PNG_BYTES
    assert result.meta["endpoint"] == "edits"
    assert result.meta["reference_images"] == [str(ref)]


def test_openai_image_requires_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    gen = OpenAIImageGen()

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        gen.generate("p", out_path=str(tmp_path / "out.png"))
