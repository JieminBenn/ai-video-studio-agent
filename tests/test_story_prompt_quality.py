"""Prompt-quality contracts for story planning stages.

These tests do not judge taste. They lock in that the planning prompts ask for
specific, filmable story data instead of generic placeholder beats, because real video
quality depends on the structured plot/script/storyboard artifacts upstream.
"""

from studio_agent import cli
from studio_agent.providers.openai_compatible import OpenAICompatibleLLM
from studio_agent.stages import plot, script, storyboard


def test_plot_prompt_demands_specific_cast_conflict_and_set_pieces():
    prompt = plot._PROMPT_TEMPLATE

    assert "not generic" in prompt.lower()
    assert "named" in prompt.lower() and "characters" in prompt.lower()
    assert "central conflict" in prompt.lower()
    assert "visual set pieces" in prompt.lower()
    assert "emotional turn" in prompt.lower()


def test_script_prompt_demands_filmable_beats_and_subtext():
    prompt = script._PROMPT_TEMPLATE

    assert "filmable" in prompt.lower()
    assert "subtext" in prompt.lower()
    assert "visual action" in prompt.lower()
    assert "avoid placeholder dialogue" in prompt.lower()
    assert "physical action" in prompt.lower()


def test_storyboard_prompt_demands_concrete_motion_camera_and_continuity():
    prompt = storyboard._PROMPT_TEMPLATE

    assert "concrete physical action" in prompt.lower()
    assert "camera movement" in prompt.lower()
    assert "composition" in prompt.lower()
    assert "continuity_notes" in prompt
    assert "emotion" in prompt.lower()


def test_deepseek_profiles_use_openai_compatible_provider():
    config = cli.load_config()

    story = config["profiles"]["deepseek-story"]
    assert story["llm"] == "openai-compatible"
    assert story["llm_base_url"] == "https://api.deepseek.com"
    assert story["llm_api_key_env"] == "DEEPSEEK_API_KEY"
    assert story["llm_model"] == "deepseek-v4-flash"
    assert story["image"] == "fake"
    assert story["video"] == "fake"

    real = config["profiles"]["real-video-intl-deepseek-qc-cheap"]
    assert real["llm"] == "openai-compatible"
    assert real["llm_api_key_env"] == "DEEPSEEK_API_KEY"
    assert real["image"] == "gemini"
    assert real["video"] == "byteplus"
    assert real["vlm"] == "gemini"

    byteplus_story = config["profiles"]["byteplus-deepseek-story"]
    assert byteplus_story["llm"] == "openai-compatible"
    assert byteplus_story["llm_base_url"] == "https://ark.ap-southeast.bytepluses.com/api/v3"
    assert byteplus_story["llm_api_key_env"] == "BYTEPLUS_ARK_API_KEY"
    assert byteplus_story["llm_model"] == "deepseek-v4-flash-260425"
    assert byteplus_story["llm_json_response_format"] is False
    assert byteplus_story["image"] == "fake"
    assert byteplus_story["video"] == "fake"

    byteplus_real = config["profiles"]["real-video-intl-byteplus-deepseek-qc-cheap"]
    assert byteplus_real["llm"] == "openai-compatible"
    assert byteplus_real["llm_base_url"] == "https://ark.ap-southeast.bytepluses.com/api/v3"
    assert byteplus_real["llm_api_key_env"] == "BYTEPLUS_ARK_API_KEY"
    assert byteplus_real["llm_model"] == "deepseek-v4-flash-260425"
    assert byteplus_real["llm_json_response_format"] is False
    assert byteplus_real["image"] == "gemini"
    assert byteplus_real["video"] == "byteplus"
    assert byteplus_real["vlm"] == "gemini"

    china_real = config["profiles"]["real-video-cn-doubao-deepseek-qc-cheap"]
    assert china_real["llm"] == "openai-compatible"
    assert china_real["llm_base_url"] == "https://ark.cn-beijing.volces.com/api/v3"
    assert china_real["llm_api_key_env"] == "ARK_API_KEY"
    assert china_real["llm_model"] == "deepseek-v4-flash-260425"
    assert china_real["llm_json_response_format"] is False
    assert china_real["image"] == "doubao"
    assert china_real["image_api_key_env"] == "ARK_API_KEY"
    assert china_real["video"] == "volcengine"
    assert china_real["video_model"] == "doubao-seedance-2-0-fast-260128"
    assert china_real["vlm"] == "openai-compatible-vision"
    assert china_real["vlm_api_key_env"] == "ARK_API_KEY"
    assert china_real["vlm_model"] == "doubao-seed-2-0-lite-260428"

    china_hq = config["profiles"]["real-video-cn-hq"]
    assert china_hq["video_model"] == "doubao-seedance-2-0-260128"
    assert china_hq["video_resolution"] == "1080p"
    assert china_hq["video_cost_per_million_tokens_cny"] == 51.0

    defaults = config["default_model_options"]
    assert defaults == {
        "llm": "volcengine-deepseek-v4-flash",
        "image": "doubao-seedream",
        "video": "china-seedance-fast",
        "vlm": "doubao-seed-2-lite-qc",
    }
    options = config["model_options"]
    assert options["llm"]["deepseek-direct"]["llm_api_key_env"] == "DEEPSEEK_API_KEY"
    assert options["llm"]["volcengine-deepseek-v4-flash"]["llm_model"] == "deepseek-v4-flash-260425"
    assert options["llm"]["volcengine-deepseek-v4-pro"]["llm_model"] == "deepseek-v4-pro-260425"
    assert options["llm"]["volcengine-deepseek-v4-pro"]["llm_cost_per_million_input"] == 12.0
    assert options["llm"]["volcengine-deepseek-v4-pro"]["llm_cost_per_million_cached_input"] == 1.0
    assert options["llm"]["volcengine-deepseek-v4-pro"]["llm_cost_per_million_output"] == 24.0
    assert options["llm"]["volcengine-seed-2-lite"]["llm_model"] == "doubao-seed-2-0-lite-260428"
    assert options["llm"]["volcengine-seed-1-6-vision"]["llm_model"] == "doubao-seed-1-6-vision-250815"
    assert options["llm"]["byteplus-seed-2-pro"]["llm_model"] == "seed-2-0-pro-260328"
    assert options["llm"]["byteplus-seed-1-6"]["llm_model"] == "seed-1-6-250915"
    assert options["image"]["openai-image-2"]["image_model"] == "gpt-image-2"
    assert options["image"]["xai-grok-imagine-image"]["image_cost_per_image"] == 0.07
    assert options["image"]["xai-grok-imagine-image"]["image_cost_per_input_image"] == 0.01
    assert options["image"]["doubao-seedream"]["image_api_key_env"] == "ARK_API_KEY"
    assert options["image"]["doubao-seedream-5"]["image_model"] == "doubao-seedream-5-0-260128"
    assert options["image"]["doubao-seedream-5-lite"]["image_model"] == "doubao-seedream-5-0-lite-260128"
    assert options["image"]["doubao-seedream-4"]["image_model"] == "doubao-seedream-4-0-250828"
    assert options["image"]["byteplus-seedream-5"]["image_model"] == "seedream-5-0-260128"
    assert options["image"]["byteplus-seedream-4"]["image_model"] == "seedream-4-0-250828"
    assert options["video"]["china-seedance-fast"]["video"] == "volcengine"
    assert options["video"]["china-seedance-hq"]["video_cost_per_million_tokens_cny"] == 51.0
    assert options["video"]["china-seedance-mini"]["video_model"] == "doubao-seedance-2-0-mini-260615"
    assert options["video"]["china-seedance-1-5-pro"]["video_model"] == "doubao-seedance-1-5-pro-251215"
    assert options["video"]["byteplus-seedance-fast"]["video"] == "byteplus"
    assert options["video"]["byteplus-seedance-hq"]["video_model"] == "dreamina-seedance-2-0-260128"
    assert options["vlm"]["xai-grok-4-3-qc"]["vlm_cost_per_million_input"] == 1.25
    assert options["vlm"]["xai-grok-4-3-qc"]["vlm_cost_per_million_output"] == 2.5
    assert options["vlm"]["anthropic-claude-opus-4-8-qc"]["vlm"] == "anthropic"
    assert options["vlm"]["doubao-seed-2-lite-qc"]["vlm_model"] == "doubao-seed-2-0-lite-260428"
    assert options["vlm"]["doubao-seed-1-6-vision-qc"]["vlm_model"] == "doubao-seed-1-6-vision-250815"
    assert options["vlm"]["byteplus-seed-2-pro-qc"]["vlm_model"] == "seed-2-0-pro-260328"


def test_cli_builds_deepseek_llm_from_profile():
    profile = {
        "llm": "openai-compatible",
        "llm_base_url": "https://api.deepseek.com",
        "llm_api_key_env": "DEEPSEEK_API_KEY",
        "llm_model": "deepseek-v4-flash",
        "image": "fake",
        "video": "fake",
        "tts": "fake",
        "music": "fake",
        "vlm": "fake",
    }

    providers = cli.build_providers(profile)

    assert isinstance(providers.llm, OpenAICompatibleLLM)
    assert providers.llm.base_url == "https://api.deepseek.com"
    assert providers.llm.model == "deepseek-v4-flash"
    assert providers.llm.api_key_env == "DEEPSEEK_API_KEY"


def test_storyboard_prompt_includes_extended_shape_for_real_llms():
    # The prompt must name every field expected from a real LLM, because the stage
    # preserves unknown keys for later prompt compilers and QC.
    expected = {
        "scene",
        "camera",
        "camera_movement",
        "action",
        "description",
        "composition",
        "emotion",
        "continuity_notes",
        "duration_s",
        "characters",
    }

    prompt = storyboard._PROMPT_TEMPLATE

    for field in expected:
        assert field in prompt
