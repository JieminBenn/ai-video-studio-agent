"""Tests for the Gemini-backed VLM QC provider.

The live Gemini video-understanding call is paid + non-deterministic, so tests inject a
``responder`` and exercise only deterministic normalization, cost calculation, wiring,
and review-stage integration. No network and no API key.
"""

import json

import pytest

from studio_agent.providers.base import QC_DIMENSIONS
from studio_agent.providers.gemini_vlm import (
    GeminiVLMCheck,
    _SCHEMA,
    _supports_config_media_resolution,
    _user_text,
    _video_part,
)


def _pass_verdict():
    return {
        "verdict": {
            "checks": [
                {
                    "dimension": dim,
                    "passed": True,
                    "severity": "none",
                    "detail": "ok",
                    "timestamp": None,
                }
                for dim in QC_DIMENSIONS
            ],
            "summary": "clean clip",
        },
        "usage": {"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500},
        "model": "gemini-2.5-flash-lite",
        "video_input_mode": "inline",
    }


def _make(response):
    return GeminiVLMCheck(responder=lambda **kw: response)


def test_gemini_vlm_describes_reference_identity(tmp_path):
    image = tmp_path / "goddess.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nref")
    identity = {
        "visual_description": "ornate fantasy goddess with a gold headdress",
        "wardrobe": "gold headdress, red-and-gold armored dress",
        "palette": "gold, crimson",
    }
    calls = []

    def responder(**kw):
        calls.append(kw)
        return {"text": json.dumps(identity), "usage": {"input_tokens": 10, "output_tokens": 5},
                "model": "gemini-x"}

    vlm = GeminiVLMCheck(reference_responder=responder)
    gen = vlm.describe([str(image)], prompt="describe this character's canonical identity")

    assert gen.content == identity
    assert calls[0]["image_paths"] == [str(image)]
    assert "canonical identity" in calls[0]["prompt"]


def test_gemini_vlm_analyzes_reference_image(tmp_path):
    image = tmp_path / "mara.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nref")
    analysis = {"target_type": "character", "target_id": "Mara", "confidence": 0.9}

    vlm = GeminiVLMCheck(
        reference_responder=lambda **kw: {"text": json.dumps(analysis), "usage": {}, "model": "g"}
    )
    gen = vlm.analyze(str(image), aliases=["@image1"], user_note="the lead")

    assert gen.content["target_type"] == "character"
    assert gen.content["target_id"] == "Mara"


def test_review_builds_a_pass_report_covering_every_dimension(tmp_path):
    clip = tmp_path / "sh-001.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftypmp42")

    gen = _make(_pass_verdict()).review(str(clip), prompt="a moonlit corridor")

    report = gen.content
    assert report["overall_pass"] is True
    assert report["recommendation"] == "approve"
    assert [c["dimension"] for c in report["checks"]] == list(QC_DIMENSIONS)
    assert report["clip"] == "sh-001.mp4"
    assert gen.provider == "gemini"
    assert gen.model == "gemini-2.5-flash-lite"


class _ApiError(Exception):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code


def test_review_retries_a_transient_server_error_then_succeeds(tmp_path):
    # Gemini returns transient 503 UNAVAILABLE under load; QC must retry rather
    # than abort the paid run at the review stage.
    clip = tmp_path / "sh-001.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    calls = {"n": 0}

    def responder(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _ApiError(503, "503 UNAVAILABLE: high demand, try again later")
        return _pass_verdict()

    gen = GeminiVLMCheck(responder=responder, max_attempts=3, retry_backoff_s=0)
    report = gen.review(str(clip), prompt="a moonlit corridor").content

    assert calls["n"] == 2  # retried once
    assert report["overall_pass"] is True


def test_review_does_not_retry_a_client_error(tmp_path):
    clip = tmp_path / "sh-001.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    calls = {"n": 0}

    def responder(**kwargs):
        calls["n"] += 1
        raise _ApiError(400, "400 INVALID_ARGUMENT")

    gen = GeminiVLMCheck(responder=responder, max_attempts=3, retry_backoff_s=0)
    with pytest.raises(_ApiError):
        gen.review(str(clip), prompt="x")
    assert calls["n"] == 1  # client errors fail fast, no retry


def test_review_flags_failure_and_recommends_regeneration(tmp_path):
    clip = tmp_path / "sh-001.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    verdict = _pass_verdict()
    verdict["verdict"]["checks"][1] = {
        "dimension": QC_DIMENSIONS[1],
        "passed": False,
        "severity": "medium",
        "detail": "the character face drifts from the reference",
        "timestamp": "00:01",
    }

    report = _make(verdict).review(str(clip), prompt="x").content

    assert report["overall_pass"] is False
    assert report["recommendation"] == "regenerate"
    assert report["max_severity"] == "medium"


def test_low_artifact_failure_is_warning_not_regeneration(tmp_path):
    clip = tmp_path / "sh-001.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    verdict = _pass_verdict()
    artifact = next(
        c for c in verdict["verdict"]["checks"] if c["dimension"] == "artifacts"
    )
    artifact.update({
        "passed": False,
        "severity": "low",
        "detail": "minor texture shimmer",
        "timestamp": "00:01",
    })

    report = _make(verdict).review(str(clip), prompt="x").content

    assert report["overall_pass"] is True
    assert report["recommendation"] == "approve_with_warnings"
    assert report["has_warnings"] is True


def test_review_fails_missing_required_dimensions(tmp_path):
    clip = tmp_path / "sh-001.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    verdict = {
        "verdict": {
            "checks": [{"dimension": QC_DIMENSIONS[0], "passed": True, "severity": "none"}],
            "summary": "partial",
        },
        "usage": {"prompt_token_count": 100, "candidates_token_count": 10},
    }

    report = _make(verdict).review(str(clip), prompt="x").content

    assert len(report["checks"]) == len(QC_DIMENSIONS)
    assert report["overall_pass"] is False
    missing = report["checks"][1]
    assert missing["passed"] is False
    assert missing["severity"] == "high"
    assert missing["detail"] == "required QC dimension missing from provider response"


def test_gemini_review_prompt_states_strict_native_sound_policy():
    text = _user_text("Mara says exactly: Where am I?")

    assert "verify exact speaker and dialogue" in text
    assert "reject invented speech" in text
    assert "synchronized diegetic SFX and natural ambience" in text
    assert "score, song, singing, beat, melody, or underscore" in text


def test_review_computes_cost_from_token_usage(tmp_path):
    clip = tmp_path / "sh-001.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftypmp42")

    gen = _make(_pass_verdict()).review(str(clip), prompt="x")

    # 1000 input @ $0.0001/1k + 500 output @ $0.0004/1k = $0.0003.
    assert gen.cost_usd == pytest.approx(0.0003)
    assert gen.meta["usage"]["input_tokens"] == 1000
    assert gen.meta["video_input_mode"] == "inline"


def test_review_raises_for_a_missing_clip(tmp_path):
    with pytest.raises(FileNotFoundError):
        _make(_pass_verdict()).review(str(tmp_path / "nope.mp4"), prompt="x")


def test_default_responder_is_lazy_no_network_on_construct():
    provider = GeminiVLMCheck()
    assert provider.model == "gemini-2.5-flash-lite"
    assert provider.name == "gemini"


def test_gemini_direct_video_qc_can_review_audio():
    assert GeminiVLMCheck(responder=lambda **_: {}).supports_audio_review is True


def test_response_schema_uses_only_gemini_supported_keywords():
    from google.genai import types

    config = types.GenerateContentConfig(
        responseMimeType="application/json",
        responseSchema=_SCHEMA,
    )
    payload = config.model_dump(by_alias=True, exclude_none=True)

    assert "additionalProperties" not in str(payload)
    assert "additional_properties" not in str(payload)


def test_inline_video_part_uses_documented_blob_shape_without_media_resolution(tmp_path):
    from google.genai import types

    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftypmp42")

    part, mode = _video_part(
        client=None,
        types=types,
        clip=clip,
        media_resolution="low",
        inline_video_max_mb=20.0,
    )
    payload = part.model_dump(by_alias=True, exclude_none=True)

    assert mode == "inline"
    assert payload["inlineData"]["mimeType"] == "video/mp4"
    assert "mediaResolution" not in str(payload)


def test_media_resolution_config_is_only_sent_for_gemini_3_models():
    assert _supports_config_media_resolution("gemini-2.5-flash-lite") is False
    assert _supports_config_media_resolution("models/gemini-2.5-flash") is False
    assert _supports_config_media_resolution("gemini-3.1-flash-lite") is True
    assert _supports_config_media_resolution("models/gemini-3.5-flash") is True


def test_build_providers_wires_the_gemini_vlm():
    from studio_agent.cli import build_providers

    providers = build_providers({
        "vlm": "gemini",
        "vlm_model": "gemini-2.5-flash-lite",
        "vlm_media_resolution": "low",
    })

    assert isinstance(providers.vlm, GeminiVLMCheck)
    assert providers.vlm.model == "gemini-2.5-flash-lite"
    assert providers.vlm.media_resolution == "low"


def test_review_stage_writes_report_via_gemini_vlm(tmp_path):
    from studio_agent.orchestrator.project import Project
    from studio_agent.stages.base import Providers
    from studio_agent.stages.review import ReviewStage

    p = Project.create("gemini review", root=tmp_path, stages=["review"])
    p.path("storyboard", "shots.json").write_text(json.dumps({
        "shots": [{"id": "sh-001", "keyframe": "sh-001.png", "description": "pier"}]
    }))
    clip = p.path("assets", "clips", "sh-001.mp4")
    clip.parent.mkdir(parents=True, exist_ok=True)
    clip.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    p.path("assets", "clips", "clips.json").write_text(
        json.dumps({"clips": [{"id": "sh-001", "clip": "sh-001.mp4"}]})
    )

    vlm = GeminiVLMCheck(responder=lambda **kw: _pass_verdict())
    result = ReviewStage().run(p, Providers(vlm=vlm))

    assert result.status == "complete"
    report = json.loads(p.path("assets", "qc", "sh-001.json").read_text())
    assert report["overall_pass"] is True
    assert "clip_hash" in report
