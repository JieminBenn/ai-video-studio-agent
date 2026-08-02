"""AnthropicStyleProfiler: deterministic core via an injected responder (no network)."""

import pytest

from studio_agent.providers.base import StyleProfiler
from studio_agent.providers.fake import FakeImageGen
from studio_agent.providers.anthropic_vlm import AnthropicStyleProfiler

REQUIRED = {"look", "palette", "aspect_ratio", "medium", "idiom", "lighting",
            "color_grade", "lens", "atmosphere", "rendering", "line_style", "motion"}


def test_is_a_style_profiler():
    assert isinstance(AnthropicStyleProfiler(responder=lambda **k: {"style": {}}), StyleProfiler)


def test_profile_from_image_normalizes_and_marks_seen(tmp_path):
    image = tmp_path / "ref.png"
    FakeImageGen().generate("style", out_path=str(image), seed=7)
    captured = {}

    def respond(*, description, image_path, language, model, max_tokens, api_key_env, feedback=""):
        captured.update(image_path=image_path, language=language, model=model)
        return {"style": {"look": "guoman 3D", "medium": "3D CG render", "idiom": "guoman"},
                "usage": {"input_tokens": 1000, "output_tokens": 200}, "model": model}

    prof = AnthropicStyleProfiler(model="claude-opus-4-8", responder=respond,
                                  cost_per_1k_input_usd=0.005, cost_per_1k_output_usd=0.025)
    gen = prof.profile(description="x", image_path=str(image), language="en")

    assert REQUIRED <= set(gen.content)
    assert gen.content["medium"] == "3D CG render"
    assert gen.provider == "anthropic-style"
    assert gen.meta["saw_image"] is True
    # 1000/1000*0.005 + 200/1000*0.025 = 0.005 + 0.005 = 0.01
    assert gen.cost_usd == pytest.approx(0.01)
    assert captured["model"] == "claude-opus-4-8"
    assert captured["language"] == "en"


def test_profile_text_only_marks_not_seen():
    prof = AnthropicStyleProfiler(responder=lambda **k: {"style": {"look": "x"}, "usage": {}, "model": "m"})
    gen = prof.profile(description="watercolor")
    assert gen.meta["saw_image"] is False
    assert "aspect_ratio" in gen.content


def test_profile_requires_description_or_image():
    prof = AnthropicStyleProfiler(responder=lambda **k: {"style": {}})
    with pytest.raises(ValueError):
        prof.profile()


def test_profile_missing_image_file_raises(tmp_path):
    prof = AnthropicStyleProfiler(responder=lambda **k: {"style": {}})
    with pytest.raises(FileNotFoundError):
        prof.profile(image_path=str(tmp_path / "nope.png"))
