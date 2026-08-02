"""Capability-aware, key-independent model catalog tests."""

from studio_agent import cli
from studio_agent.providers.base import ImageGen, LLM, VideoGen, VLMCheck
from studio_agent.web import (
    _compose_model_mix,
    _model_option_select,
    render_index,
)


def _models(config, kind, field):
    return {
        option.get(field)
        for option in config["model_options"][kind].values()
        if option.get(field)
    }


def test_dashboard_is_preset_free_and_has_four_independent_selectors(tmp_path):
    html = render_index(tmp_path)

    # The dashboard offers the user-facing generation selectors, including the vision
    # model selector added to expose VLM choice for reference analysis and style extraction.
    for kind in ("llm", "image", "video", "vlm"):
        assert (
            f"<select name='{kind}_option' data-model-preference='{kind}'>" in html
        )
    assert "Vision model" in html
    assert "Run full pipeline" in html
    assert "Cheap test run" not in html
    assert "__cheap_test__" not in html
    assert "name='profile'" not in html
    assert "<th>Profile</th>" not in html
    assert html.count("type='submit'") >= 1


def test_catalog_contains_major_current_provider_families():
    config = cli.load_config()

    llms = _models(config, "llm", "llm_model")
    assert {
        "gpt-5.5", "claude-opus-4-8", "gemini-3.5-flash", "grok-4.3",
        "grok-4.20-multi-agent-0309", "grok-4.20-0309-reasoning",
        "grok-4.20-0309-non-reasoning",
    } <= llms

    images = _models(config, "image", "image_model")
    assert {
        "gpt-image-2",
        "gemini-3.1-flash-image",
        "gemini-3-pro-image",
        "gemini-2.5-flash-image",
        "grok-imagine-image-quality",
    } <= images

    videos = _models(config, "video", "video_model")
    assert {
        "veo-3.1-generate-preview",
        "veo-3.1-fast-generate-preview",
        "veo-3.1-lite-generate-preview",
        "grok-imagine-video",
        "grok-imagine-video-1.5",
    } <= videos

    vlms = _models(config, "vlm", "vlm_model")
    assert {"gpt-5.4-mini", "claude-opus-4-8", "gemini-3.5-flash", "grok-4.3"} <= vlms


def test_model_options_show_missing_key_without_filtering(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = {
        "model_options": {
            "llm": {
                "gpt": {
                    "label": "GPT",
                    "group": "OpenAI",
                    "requires": ["OPENAI_API_KEY"],
                    "llm_model": "gpt-5.5",
                }
            }
        }
    }

    html = _model_option_select(config, "llm")

    assert "value='gpt'" in html
    assert "needs OPENAI_API_KEY" in html
    assert "disabled" not in html


def test_model_options_show_ready_without_leaking_secret(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "xai-super-secret")
    config = {
        "model_options": {
            "llm": {
                "grok": {
                    "label": "Grok",
                    "group": "xAI",
                    "requires": ["XAI_API_KEY"],
                    "availability": "preview",
                    "llm_model": "grok-4.3",
                }
            }
        }
    }

    html = _model_option_select(config, "llm")

    assert "preview" in html
    assert "ready" in html
    assert "xai-super-secret" not in html


def test_cross_provider_mix_is_exact_and_metadata_does_not_leak():
    config = cli.load_config()
    selected = {
        "llm": "openai-gpt-5-5",
        "image": "gemini",
        "video": "xai-grok-imagine-video",
        "vlm": "anthropic-claude-opus-4-8-qc",
    }

    profile = _compose_model_mix(config, selected)

    assert profile["llm_model"] == "gpt-5.5"
    assert profile["image_model"] == "gemini-3.1-flash-image"
    assert profile["video_model"] == "grok-imagine-video"
    assert profile["vlm_model"] == "claude-opus-4-8"
    assert not ({"label", "group", "provider", "availability", "requires", "capabilities"} & profile.keys())


def test_every_media_and_qc_option_constructs_lazily_without_keys(monkeypatch):
    config = cli.load_config()
    for key in (
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "XAI_API_KEY",
        "ARK_API_KEY", "BYTEPLUS_ARK_API_KEY", "FAL_KEY",
    ):
        monkeypatch.delenv(key, raising=False)

    defaults = config["default_model_options"]
    for kind, expected_type, provider_attr in (
        ("llm", LLM, "llm"),
        ("image", ImageGen, "image"),
        ("video", VideoGen, "video"),
        ("vlm", VLMCheck, "vlm"),
    ):
        for option_id in config["model_options"][kind]:
            requested = dict(defaults)
            requested[kind] = option_id
            profile = _compose_model_mix(config, requested)
            provider = getattr(cli.build_providers(profile), provider_attr)
            assert isinstance(provider, expected_type), (kind, option_id, provider)
