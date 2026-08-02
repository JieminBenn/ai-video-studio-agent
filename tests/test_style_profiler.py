"""Deterministic style profiler: free text / uploaded image -> reusable style dict.

The profiler turns whatever the user gives (a short label, a sentence, or an
uploaded reference image) into the *same* style-dict shape the fixed presets used,
so every downstream prompt threads the custom style with no other change. The fake
runs offline with no network or cost so the whole pipeline still runs on fakes.
"""

from studio_agent.providers.base import StyleProfiler
from studio_agent.providers.fake import FakeImageGen, FakeStyleProfiler

# Keys the rest of the pipeline reads off a style dict (style.py / format_style_*).
REQUIRED_STYLE_KEYS = {
    "look", "palette", "aspect_ratio", "rendering", "line_style", "motion", "prompt_playbook",
}


def test_fake_style_profiler_is_a_style_profiler():
    assert isinstance(FakeStyleProfiler(), StyleProfiler)


def test_profile_from_text_returns_full_style_dict_without_network():
    gen = FakeStyleProfiler().profile(description="1970s grainy analog sci-fi")

    assert gen.provider == "fake"
    assert gen.cost_usd == 0.0
    style = gen.content
    assert REQUIRED_STYLE_KEYS <= set(style)
    # The director needs a non-empty playbook that echoes the user's words.
    assert style["prompt_playbook"].strip()
    assert "sci-fi" in style["prompt_playbook"].lower()
    # A short human-facing label is produced for the gate / bible.
    assert style["label"].strip()
    assert style["aspect_ratio"] == "16:9"


def test_profile_from_text_is_deterministic():
    a = FakeStyleProfiler().profile(description="dreamy pastel watercolor")
    b = FakeStyleProfiler().profile(description="dreamy pastel watercolor")
    assert a.content == b.content


def test_profile_from_image_derives_style_only_from_the_reference(tmp_path):
    image = tmp_path / "moody-noir-style.png"
    FakeImageGen().generate("style", out_path=str(image), seed=7)

    gen = FakeStyleProfiler().profile(image_path=str(image))

    style = gen.content
    assert REQUIRED_STYLE_KEYS <= set(style)
    # Derived from the uploaded reference, labelled from its filename.
    assert "noir" in style["label"].lower()
    assert style["prompt_playbook"].strip()


def test_build_providers_wires_fake_style_profiler():
    from studio_agent.cli import build_providers

    providers = build_providers({"llm": "fake"})

    assert isinstance(providers.style_profiler, FakeStyleProfiler)


def test_profile_with_text_and_image_lets_text_refine_the_look(tmp_path):
    image = tmp_path / "reference.png"
    FakeImageGen().generate("style", out_path=str(image), seed=8)

    gen = FakeStyleProfiler().profile(
        description="high-contrast ink comic", image_path=str(image)
    )

    blob = (gen.content["look"] + gen.content["prompt_playbook"]).lower()
    assert "ink comic" in blob


def test_wire_uses_vision_profiler_when_vlm_is_openai_compatible():
    from studio_agent.cli import _wire_style_profiler
    from studio_agent.stages.base import Providers
    from studio_agent.providers.openai_compatible_vlm import OpenAICompatibleStyleProfiler

    providers = Providers()
    _wire_style_profiler(providers, {
        "vlm": "doubao-vlm",
        "vlm_base_url": "https://example/api/v3",
        "vlm_model": "doubao-seed",
        "vlm_api_key_env": "ARK_API_KEY",
    })
    assert isinstance(providers.style_profiler, OpenAICompatibleStyleProfiler)
    assert providers.style_profiler.model == "doubao-seed"
    assert providers.style_profiler.base_url == "https://example/api/v3"


def test_wire_falls_back_to_text_profiler_without_a_vision_vlm():
    from studio_agent.cli import _wire_style_profiler
    from studio_agent.stages.base import Providers
    from studio_agent.providers.fake import FakeLLM
    from studio_agent.providers.llm_style import LLMStyleProfiler

    providers = Providers(llm=FakeLLM())
    _wire_style_profiler(providers, {"vlm": "some-unknown-vlm"})  # unknown vlm has no style profiler
    assert isinstance(providers.style_profiler, LLMStyleProfiler)


def test_wire_uses_gemini_style_profiler_for_gemini_vlm():
    from studio_agent.cli import _wire_style_profiler
    from studio_agent.stages.base import Providers
    from studio_agent.providers.gemini_vlm import GeminiStyleProfiler

    providers = Providers()
    _wire_style_profiler(providers, {"vlm": "gemini", "vlm_model": "gemini-3.5-flash"})
    assert isinstance(providers.style_profiler, GeminiStyleProfiler)
    assert providers.style_profiler.model == "gemini-3.5-flash"


def test_wire_uses_anthropic_style_profiler_for_anthropic_vlm():
    from studio_agent.cli import _wire_style_profiler
    from studio_agent.stages.base import Providers
    from studio_agent.providers.anthropic_vlm import AnthropicStyleProfiler

    providers = Providers()
    _wire_style_profiler(providers, {"vlm": "anthropic", "vlm_model": "claude-opus-4-8"})
    assert isinstance(providers.style_profiler, AnthropicStyleProfiler)
    assert providers.style_profiler.model == "claude-opus-4-8"


def test_fake_profiler_applies_feedback_idempotently():
    base = FakeStyleProfiler().profile(description="moody analog sci-fi")
    warm = FakeStyleProfiler().profile(description="moody analog sci-fi", feedback="make it warmer")
    warm2 = FakeStyleProfiler().profile(description="moody analog sci-fi", feedback="make it warmer")
    assert "warmer" in (warm.content["look"] + warm.content["prompt_playbook"]).lower()
    assert warm.content != base.content
    assert warm.content == warm2.content  # idempotent


def test_llm_style_profiler_includes_feedback_in_prompt():
    from studio_agent.providers.base import Generation
    from studio_agent.providers.llm_style import LLMStyleProfiler

    seen = {}

    class _RecLLM:
        provider = "rec"
        def complete_json(self, prompt, *, system=None):
            seen["prompt"] = prompt
            return Generation(content={"look": "x", "label": "X"}, provider="rec",
                              model="m", cost_usd=0.0, seconds=0.0, meta={})

    LLMStyleProfiler(_RecLLM()).profile(description="moody", feedback="make it warmer")
    assert "warmer" in seen["prompt"].lower()
