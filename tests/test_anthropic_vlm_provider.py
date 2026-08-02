"""Tests for the real (Claude-backed) VLM QC provider.

Deterministic only: the non-deterministic, paid model call is injected as a ``responder``
so the report-normalization and cost logic are exercised offline. No network, no key, no
ffmpeg. The lazy default responder (frame sampling + Anthropic SDK) is only hit in paid
manual smoke tests.
"""

import json

import pytest

from studio_agent.providers.anthropic_vlm import AnthropicVLMCheck
from studio_agent.providers.base import QC_DIMENSIONS


def _pass_verdict():
    return {
        "verdict": {
            "checks": [
                {"dimension": d, "passed": True, "severity": "none", "detail": "ok", "timestamp": None}
                for d in QC_DIMENSIONS
            ],
            "summary": "clean clip",
        },
        "usage": {"input_tokens": 2000, "output_tokens": 400},
        "model": "claude-opus-4-8",
    }


def _make(responder):
    return AnthropicVLMCheck(responder=lambda **kw: responder)


def test_anthropic_vlm_describes_reference_identity(tmp_path):
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
                "model": "claude-x"}

    vlm = AnthropicVLMCheck(reference_responder=responder)
    gen = vlm.describe([str(image)], prompt="describe this character's canonical identity")

    assert gen.content == identity
    assert calls[0]["image_paths"] == [str(image)]
    assert "canonical identity" in calls[0]["prompt"]


def test_anthropic_vlm_analyzes_reference_image(tmp_path):
    image = tmp_path / "mara.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nref")
    analysis = {"target_type": "character", "target_id": "Mara", "confidence": 0.9}

    vlm = AnthropicVLMCheck(
        reference_responder=lambda **kw: {"text": json.dumps(analysis), "usage": {}, "model": "c"}
    )
    gen = vlm.analyze(str(image), aliases=["@image1"], user_note="the lead")

    assert gen.content["target_type"] == "character"
    assert gen.content["target_id"] == "Mara"


def test_review_builds_a_pass_report_covering_every_dimension(tmp_path):
    clip = tmp_path / "sh-001.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftypmp42")

    gen = _make(_pass_verdict()).review(str(clip), prompt="a windswept pier at dawn")

    report = gen.content
    assert report["overall_pass"] is True
    assert report["recommendation"] == "approve"
    assert [c["dimension"] for c in report["checks"]] == list(QC_DIMENSIONS)
    assert report["clip"] == "sh-001.mp4"
    assert gen.provider == "anthropic"


def test_review_flags_failure_and_recommends_regeneration(tmp_path):
    clip = tmp_path / "sh-001.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    verdict = _pass_verdict()
    verdict["verdict"]["checks"][2] = {
        "dimension": QC_DIMENSIONS[2], "passed": False, "severity": "high",
        "detail": "character morphs mid-clip", "timestamp": "00:02",
    }

    report = _make(verdict).review(str(clip), prompt="x").content

    assert report["overall_pass"] is False
    assert report["recommendation"] == "regenerate"
    assert report["max_severity"] == "high"


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
        "detail": "minor shimmer",
        "timestamp": "00:01",
    })

    report = _make(verdict).review(str(clip), prompt="x").content

    assert report["overall_pass"] is True
    assert report["recommendation"] == "approve_with_warnings"
    assert report["has_warnings"] is True


def test_review_fills_missing_dimensions_as_passed(tmp_path):
    clip = tmp_path / "sh-001.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    verdict = {
        "verdict": {"checks": [
            {"dimension": QC_DIMENSIONS[0], "passed": True, "severity": "none"}
        ], "summary": "partial"},
        "usage": {"input_tokens": 100, "output_tokens": 10},
        "model": "claude-opus-4-8",
    }

    report = _make(verdict).review(str(clip), prompt="x").content

    assert len(report["checks"]) == len(QC_DIMENSIONS)
    assert report["overall_pass"] is True


def test_review_computes_cost_from_token_usage(tmp_path):
    clip = tmp_path / "sh-001.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftypmp42")

    gen = _make(_pass_verdict()).review(str(clip), prompt="x")

    # 2000 input @ $0.005/1k + 400 output @ $0.025/1k = 0.010 + 0.010 = 0.020
    assert gen.cost_usd == pytest.approx(0.02)
    assert gen.meta["usage"]["input_tokens"] == 2000


def test_review_raises_for_a_missing_clip(tmp_path):
    with pytest.raises(FileNotFoundError):
        _make(_pass_verdict()).review(str(tmp_path / "nope.mp4"), prompt="x")


def test_default_responder_is_lazy_no_network_on_construct():
    # Constructing the provider must not import the SDK or touch the network/key.
    provider = AnthropicVLMCheck()
    assert provider.model == "claude-opus-4-8"
    assert provider.name == "anthropic"


def test_build_providers_wires_the_anthropic_vlm():
    from studio_agent.cli import build_providers

    providers = build_providers({"vlm": "anthropic"})
    assert isinstance(providers.vlm, AnthropicVLMCheck)


def test_review_stage_writes_report_via_anthropic_vlm(tmp_path):
    from studio_agent.orchestrator.project import Project
    from studio_agent.stages.base import Providers
    from studio_agent.stages.review import ReviewStage

    p = Project.create("vlm review", root=tmp_path, stages=["review"])
    p.path("storyboard", "shots.json").write_text(json.dumps({"shots": [
        {"id": "sh-001", "keyframe": "sh-001.png", "description": "pier"}
    ]}))
    clip = p.path("assets", "clips", "sh-001.mp4")
    clip.parent.mkdir(parents=True, exist_ok=True)
    clip.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    p.path("assets", "clips", "clips.json").write_text(
        json.dumps({"clips": [{"id": "sh-001", "clip": "sh-001.mp4"}]})
    )

    vlm = AnthropicVLMCheck(responder=lambda **kw: _pass_verdict())
    result = ReviewStage().run(p, Providers(vlm=vlm))

    assert result.status == "complete"
    report = json.loads(p.path("assets", "qc", "sh-001.json").read_text())
    assert report["overall_pass"] is True
    assert "clip_hash" in report  # the review stage stamped it
