"""GeminiStyleProfiler: deterministic core via an injected responder (no network)."""

import pytest

from studio_agent.providers.base import StyleProfiler
from studio_agent.providers.fake import FakeImageGen
from studio_agent.providers.gemini_vlm import GeminiStyleProfiler

REQUIRED = {"look", "palette", "aspect_ratio", "medium", "idiom", "lighting",
            "color_grade", "lens", "atmosphere", "rendering", "line_style", "motion"}


def _responder(captured):
    def respond(*, description, image_path, language, model, max_tokens, api_key_env, feedback=""):
        captured.update(description=description, image_path=image_path, language=language,
                        model=model)
        return {
            "style": {"look": "guoman 3D", "label": "Guoman", "medium": "3D CG render",
                      "idiom": "guoman", "lighting": "rim + bloom"},
            "usage": {"input_tokens": 1000, "output_tokens": 500},
            "model": model,
        }
    return respond


def test_is_a_style_profiler():
    assert isinstance(GeminiStyleProfiler(responder=lambda **k: {"style": {}}), StyleProfiler)


def test_profile_from_image_normalizes_and_marks_seen(tmp_path):
    image = tmp_path / "ref.png"
    FakeImageGen().generate("style", out_path=str(image), seed=7)
    captured = {}
    prof = GeminiStyleProfiler(model="gemini-3.5-flash", responder=_responder(captured),
                               cost_per_1k_input_usd=0.0001, cost_per_1k_output_usd=0.0004)

    gen = prof.profile(description="anything", image_path=str(image), language="zh")

    assert REQUIRED <= set(gen.content)
    assert gen.content["medium"] == "3D CG render"
    assert gen.content["idiom"] == "guoman"
    assert gen.provider == "gemini-style"
    assert gen.meta["saw_image"] is True
    # 1000/1000*0.0001 + 500/1000*0.0004 = 0.0001 + 0.0002 = 0.0003
    assert gen.cost_usd == pytest.approx(0.0003)
    assert captured["image_path"] == str(image)
    assert captured["language"] == "zh"


def test_profile_text_only_marks_not_seen():
    prof = GeminiStyleProfiler(responder=lambda **k: {"style": {"look": "x"}, "usage": {}, "model": "m"})
    gen = prof.profile(description="watercolor fairyland")
    assert gen.meta["saw_image"] is False
    assert "aspect_ratio" in gen.content  # normalized


def test_profile_requires_description_or_image():
    prof = GeminiStyleProfiler(responder=lambda **k: {"style": {}})
    with pytest.raises(ValueError):
        prof.profile()


def test_profile_missing_image_file_raises(tmp_path):
    prof = GeminiStyleProfiler(responder=lambda **k: {"style": {}})
    with pytest.raises(FileNotFoundError):
        prof.profile(image_path=str(tmp_path / "nope.png"))
