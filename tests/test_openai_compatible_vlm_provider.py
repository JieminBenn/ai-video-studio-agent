"""Tests for OpenAI-compatible vision QC providers without network calls."""

import json
from pathlib import Path

from studio_agent import cli
from studio_agent.providers.base import QC_DIMENSIONS
from studio_agent.providers.openai_compatible_vlm import (
    OpenAICompatibleStyleProfiler,
    OpenAICompatibleVLMCheck,
)


def _verdict(**overrides):
    checks = [
        {
            "dimension": dim,
            "passed": True,
            "severity": "none",
            "detail": f"{dim} ok",
            "timestamp": None,
        }
        for dim in QC_DIMENSIONS
    ]
    verdict = {"summary": "looks coherent", "checks": checks}
    verdict.update(overrides)
    return verdict


def _response(content, *, usage=None, model="doubao-seed-2-0-lite-260428"):
    return {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": usage or {"prompt_tokens": 100, "completion_tokens": 25},
        "model": model,
    }


def test_openai_compatible_vlm_sends_sampled_frames_as_image_parts(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake mp4")
    frame_a = tmp_path / "frame-a.png"
    frame_b = tmp_path / "frame-b.png"
    frame_a.write_bytes(b"frame a")
    frame_b.write_bytes(b"frame b")
    calls = []

    def frame_sampler(clip_path, out_dir, count):
        assert clip_path == str(clip)
        assert count == 2
        assert Path(out_dir).is_dir()
        return [str(frame_a), str(frame_b)]

    def transport(url, payload, headers, timeout_s):
        calls.append((url, payload, headers, timeout_s))
        return _response(json.dumps(_verdict()))

    vlm = OpenAICompatibleVLMCheck(
        provider_name="doubao-vlm",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        model="doubao-seed-2-0-lite-260428",
        api_key="secret",
        max_frames=2,
        detail="low",
        transport=transport,
        frame_sampler=frame_sampler,
    )

    gen = vlm.review(str(clip), prompt="story-aware QC context")

    assert gen.provider == "doubao-vlm"
    assert gen.model == "doubao-seed-2-0-lite-260428"
    assert gen.content["overall_pass"] is True
    assert gen.content["summary"] == "looks coherent"
    url, payload, headers, timeout_s = calls[0]
    assert url == "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
    assert headers["Authorization"] == "Bearer secret"
    assert timeout_s == 120.0
    content = payload["messages"][-1]["content"]
    assert [part["type"] for part in content] == ["image_url", "image_url", "text"]
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[0]["image_url"]["detail"] == "low"
    assert "story-aware QC context" in content[-1]["text"]
    assert gen.meta["frames_sampled"] == 2


def test_openai_compatible_vlm_normalizes_failed_dimensions(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake mp4")
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"frame")
    verdict = _verdict(checks=[
        {
            "dimension": "identity_drift",
            "passed": False,
            "severity": "high",
            "detail": "main character face changed",
            "timestamp": "00:02",
        }
    ])

    vlm = OpenAICompatibleVLMCheck(
        base_url="http://localhost:8000/v1",
        model="vision-local",
        transport=lambda *args: _response(f"```json\n{json.dumps(verdict)}\n```"),
        frame_sampler=lambda *args: [str(frame)],
    )

    gen = vlm.review(str(clip), prompt="qc")

    assert gen.content["overall_pass"] is False
    assert gen.content["recommendation"] == "regenerate"
    assert gen.content["blocking_failures"] == ["identity_drift"]
    assert len(gen.content["checks"]) == len(QC_DIMENSIONS)


def test_openai_compatible_vlm_analyzes_reference_image(tmp_path):
    image = tmp_path / "mara.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nreference")
    calls = []
    analysis = {
        "target_type": "character",
        "target_id": "Mara",
        "confidence": 0.94,
        "reason": "the image shows one recurring lead character",
        "visual_summary": "A woman in a red coat",
    }

    def transport(url, payload, headers, timeout_s):
        calls.append((url, payload, headers, timeout_s))
        return _response(json.dumps(analysis))

    vlm = OpenAICompatibleVLMCheck(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        model="doubao-seed-2-0-lite-260428",
        api_key="secret",
        detail="low",
        transport=transport,
    )

    gen = vlm.analyze(
        str(image),
        aliases=["@photo1", "first image"],
        user_note="first image is a lead",
    )

    assert gen.content == analysis
    url, payload, headers, timeout_s = calls[0]
    assert url.endswith("/chat/completions")
    assert headers["Authorization"] == "Bearer secret"
    content = payload["messages"][-1]["content"]
    assert [part["type"] for part in content] == ["image_url", "text"]
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "@photo1" in content[-1]["text"]
    assert "first image is a lead" in content[-1]["text"]


def test_openai_compatible_vlm_retries_connection_reset_then_succeeds(tmp_path):
    # A transient RemoteDisconnected (ConnectionResetError) must be retried, not abort the
    # bible/style stage with the raw "Remote end closed connection without response".
    image = tmp_path / "ref.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nreference")
    attempts = []
    analysis = {
        "target_type": "character", "target_id": "Mara", "confidence": 0.9,
        "reason": "r", "visual_summary": "s",
    }

    def transport(url, payload, headers, timeout_s):
        attempts.append(1)
        if len(attempts) == 1:
            raise ConnectionResetError("Remote end closed connection without response")
        return _response(json.dumps(analysis))

    vlm = OpenAICompatibleVLMCheck(
        api_key="secret", transport=transport, max_retries=3, retry_backoff_s=0.0,
    )

    gen = vlm.analyze(str(image))

    assert gen.content == analysis
    assert len(attempts) == 2


def test_grok_vlm_tracks_official_usd_token_rates(tmp_path):
    image = tmp_path / "reference.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nreference")
    analysis = {
        "target_type": "style",
        "target_id": "global",
        "confidence": 0.9,
        "reason": "global look",
        "visual_summary": "cool neon palette",
    }
    response = _response(
        json.dumps(analysis),
        usage={"prompt_tokens": 1_000_000, "completion_tokens": 500_000},
        model="grok-4.3",
    )
    vlm = OpenAICompatibleVLMCheck(
        base_url="https://api.x.ai/v1",
        model="grok-4.3",
        api_key="secret",
        transport=lambda *args: response,
        cost_per_million_input=1.25,
        cost_per_million_cached_input=1.25,
        cost_per_million_output=2.50,
        cost_currency="USD",
        pricing_source="official",
        pricing_as_of="2026-07-01",
    )

    gen = vlm.analyze(str(image))

    assert gen.cost_usd == 2.5
    assert gen.meta["cost_tracking"]["native_currency"] == "USD"
    assert gen.meta["cost_tracking"]["usage"]["input_tokens"] == 1_000_000


def test_cli_passes_official_usd_pricing_to_openai_compatible_vlm():
    providers = cli.build_providers({
        "llm": "fake",
        "image": "fake",
        "video": "fake",
        "tts": "fake",
        "music": "fake",
        "vlm": "openai-compatible-vision",
        "vlm_model": "grok-4.3",
        "vlm_cost_per_million_input": 1.25,
        "vlm_cost_per_million_cached_input": 1.25,
        "vlm_cost_per_million_output": 2.5,
        "vlm_cost_currency": "USD",
        "vlm_usd_per_native_unit": 1.0,
        "vlm_pricing_source": "official",
        "vlm_pricing_as_of": "2026-07-01",
    })

    assert providers.vlm.cost_per_million_input == 1.25
    assert providers.vlm.cost_per_million_output == 2.5
    assert providers.vlm.pricing_source == "official"


def test_openai_compatible_vlm_describe_does_not_use_video_qc_prompt(tmp_path):
    image = tmp_path / "goddess.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nreference")
    calls = []
    identity = {"visual_description": "gold-crowned fantasy woman"}

    def transport(url, payload, headers, timeout_s):
        calls.append(payload)
        return _response(json.dumps(identity))

    vlm = OpenAICompatibleVLMCheck(
        api_key="secret",
        json_response_format=True,
        transport=transport,
    )
    result = vlm.describe([str(image)], prompt="Return visual_description as JSON.")

    assert result.content == identity
    payload = calls[0]
    assert "visual reference" in payload["messages"][0]["content"].lower()
    text = payload["messages"][-1]["content"][-1]["text"]
    assert text == "Return visual_description as JSON."
    assert "Review every dimension" not in text
    assert '"summary"' not in text
    assert payload["response_format"] == {"type": "json_object"}


def test_openai_compatible_vlm_revise_does_not_force_qc_or_json_mode(tmp_path):
    image = tmp_path / "room.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nreference")
    calls = []

    def transport(url, payload, headers, timeout_s):
        calls.append(payload)
        return _response("revised prompt text")

    vlm = OpenAICompatibleVLMCheck(
        api_key="secret", json_response_format=True, transport=transport
    )
    result = vlm.revise([str(image)], prompt="Keep this room's brass windows.")

    assert result.content == "revised prompt text"
    payload = calls[0]
    assert "response_format" not in payload
    text = payload["messages"][-1]["content"][-1]["text"]
    assert text == "Keep this room's brass windows."
    assert "video shot" not in text
    assert "identity_drift" not in text


def test_style_profiler_extracts_style_from_text():
    style = {
        "look": "1970s analog sci-fi",
        "label": "70s analog sci-fi",
        "palette": "amber and teal",
        "aspect_ratio": "2.39:1",
        "rendering": "grainy film",
        "line_style": "n/a",
        "motion": "deliberate",
        "notes": "halation",
        "prompt_playbook": "Shoot on grainy 35mm anamorphic with amber/teal grade...",
    }
    calls = []

    def transport(url, payload, headers, timeout_s):
        calls.append(payload)
        return _response(json.dumps(style))

    profiler = OpenAICompatibleStyleProfiler(api_key="secret", transport=transport)
    gen = profiler.profile(description="1970s grainy analog sci-fi")

    assert gen.content["look"] == "1970s analog sci-fi"
    assert gen.content["prompt_playbook"].strip()
    # Text-only: no image part is sent.
    assert [p["type"] for p in calls[0]["messages"][-1]["content"]] == ["text"]


def test_style_profiler_sends_image_with_style_only_instruction(tmp_path):
    image = tmp_path / "noir.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nref")
    calls = []

    def transport(url, payload, headers, timeout_s):
        calls.append(payload)
        return _response(json.dumps({"look": "noir", "label": "noir"}))

    profiler = OpenAICompatibleStyleProfiler(api_key="secret", transport=transport)
    gen = profiler.profile(image_path=str(image))

    content = calls[0]["messages"][-1]["content"]
    assert [p["type"] for p in content] == ["image_url", "text"]
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    # Style-only: the prompt forbids reusing the reference's content.
    assert "never describe or reuse its subjects" in content[-1]["text"]
    # Missing keys are backfilled so downstream threading never breaks.
    assert gen.content["aspect_ratio"] == "16:9"


def test_cli_wires_reference_analyzer_for_every_vision_vlm():
    # Every vision-capable VLM must double as the reference analyzer, so uploaded images
    # ground the bible regardless of which provider the user selected.
    from studio_agent.providers.base import ReferenceAnalyzer

    base = {"llm": "fake", "image": "fake", "video": "fake", "tts": "fake", "music": "fake"}
    for vlm in ("openai-compatible-vision", "anthropic", "gemini"):
        providers = cli.build_providers({**base, "vlm": vlm})
        assert isinstance(providers.vlm, ReferenceAnalyzer), vlm
        assert providers.reference_analyzer is providers.vlm, vlm


def test_cli_builds_openai_compatible_vlm_profile():
    providers = cli.build_providers({
        "llm": "fake",
        "image": "fake",
        "video": "fake",
        "tts": "fake",
        "music": "fake",
        "vlm": "openai-compatible-vision",
        "vlm_base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "vlm_api_key_env": "ARK_API_KEY",
        "vlm_model": "doubao-seed-2-0-lite-260428",
        "vlm_provider_name": "doubao-vlm",
        "vlm_max_frames": 3,
        "vlm_detail": "high",
    })

    assert isinstance(providers.vlm, OpenAICompatibleVLMCheck)
    assert providers.vlm.base_url == "https://ark.cn-beijing.volces.com/api/v3"
    assert providers.vlm.api_key_env == "ARK_API_KEY"
    assert providers.vlm.model == "doubao-seed-2-0-lite-260428"
    assert providers.vlm.name == "doubao-vlm"
    assert providers.vlm.max_frames == 3
    assert providers.vlm.detail == "high"
    assert providers.reference_analyzer is providers.vlm


def test_cli_builds_openai_compatible_style_profiler():
    providers = cli.build_providers({
        "llm": "fake",
        "style_profiler": "openai-compatible-vision",
        "style_profiler_base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "style_profiler_api_key_env": "ARK_API_KEY",
        "style_profiler_model": "doubao-seed-2-0-lite-260428",
    })

    assert isinstance(providers.style_profiler, OpenAICompatibleStyleProfiler)
    assert providers.style_profiler.base_url == "https://ark.cn-beijing.volces.com/api/v3"
    assert providers.style_profiler.api_key_env == "ARK_API_KEY"
    assert providers.style_profiler.model == "doubao-seed-2-0-lite-260428"


def test_style_prompt_requests_new_dimensions_and_image_authority():
    from studio_agent.providers.openai_compatible_vlm import OpenAICompatibleStyleProfiler

    captured = {}

    def transport(url, payload, headers, timeout):
        captured["payload"] = payload
        return {"choices": [{"message": {"content": "{\"look\": \"x\"}"}}], "model": "m"}

    profiler = OpenAICompatibleStyleProfiler(transport=transport, api_key="k")
    profiler.profile(description="watercolor fairyland")  # text-only path is enough to read the prompt

    text = captured["payload"]["messages"][-1]["content"][-1]["text"]
    for key in ("medium", "idiom", "lighting", "color_grade", "lens", "atmosphere"):
        assert key in text
    system = captured["payload"]["messages"][0]["content"]
    assert "authoritative" in system.lower()  # image-authority conflict rule present
