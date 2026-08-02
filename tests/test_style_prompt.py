"""The style system prompt + user-message JSON shape are shared by every vision profiler."""

from studio_agent.providers.style_prompt import STYLE_SYSTEM, style_user_text

NEW_KEYS = ("medium", "idiom", "lighting", "color_grade", "lens", "atmosphere")


def test_system_prompt_states_image_is_authoritative():
    assert "authoritative" in STYLE_SYSTEM.lower()
    assert "ignore" in STYLE_SYSTEM.lower()  # ignore subjects/composition


def test_user_text_requests_every_field_and_language():
    text = style_user_text("a watercolor look", has_image=True, language="zh")
    for key in NEW_KEYS:
        assert key in text
    assert "language code 'zh'" in text
    assert "prompt_playbook" in text


def test_user_text_without_image_omits_the_image_clause():
    text = style_user_text("a watercolor look", has_image=False, language="en")
    assert "reference image is attached" not in text
    assert "watercolor look" in text


def test_openai_profiler_still_uses_the_shared_prompt():
    # The shared symbols are what the OpenAI-compatible profiler sends.
    from studio_agent.providers import openai_compatible_vlm as mod
    assert mod.STYLE_SYSTEM is STYLE_SYSTEM
