"""Command-line entry point.

    python -m studio_agent.cli run --idea "<one line>" --fake
    python -m studio_agent.cli resume <project-id>
    python -m studio_agent.cli approve <project-id>

The pipeline registry runs the implemented M0 stages. ``--fake`` selects the offline,
zero-cost provider profile.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

from . import env as env_support
from .asset_regeneration import regenerate_shot_video
from .creative_decisions import resolve_decision
from .formats import UnknownFormatError, apply_length, format_mode, resolve_format
from .language import UnknownLanguageError, resolve_language
from .knowledge.importer import import_knowledge_file, rebuild_index
from .invalidation import refresh_knowledge_target
from .orchestrator.project import CostCapExceeded, Project
from .orchestrator.state_machine import StateMachine
from .providers.base import ReferenceAnalyzer
from .providers.fake import (
    FakeImageGen, FakeLLM, FakeMusic, FakeReferenceAnalyzer, FakeStyleProfiler,
    FakeTTS, FakeVideoGen, FakeVLMCheck,
)
from .stages.assemble import AssembleStage
from .stages.audio import AudioStage
from .stages.base import Providers
from .stages.bible import BibleStage
from .stages.clip import ClipStage
from .stages.concept import ConceptStage
from .stages.keyframes import KeyframesStage
from .stages.plot import PlotStage
from .stages.script import ScriptStage
from .stages.review import ReviewStage
from .stages.storyboard import StoryboardStage
from .stages.style import StyleStage
from .stages.video import VideoStage
from .stages.video_prompts import VideoPromptsStage
from .reference_assets import save_reference_upload
from .style import CUSTOM_STYLE_NAME, UnknownStyleError, resolve_style

CONFIG_PATH = Path(__file__).with_name("config.yaml")
PROJECTS_ROOT = Path("projects")
KNOWLEDGE_ROOT = Path("knowledge")

# Story-mode stages, in pipeline order. The style stage runs first so the project-wide
# look is profiled and human-approved before anything is drawn in it.
STAGES = [
    StyleStage(), PlotStage(), ScriptStage(), BibleStage(), StoryboardStage(),
    KeyframesStage(), VideoPromptsStage(), VideoStage(), AudioStage(), AssembleStage(),
]
PIPELINE = [s.name for s in STAGES]

# Clip-mode (short_video) stages and ordered pipeline. The bible/video/audio/assemble
# stages are shared with story mode; concept+clip replace plot/script/storyboard.
CLIP_PIPELINE = [
    "style", "concept", "bible", "clip", "keyframes", "video_prompts",
    "video", "audio", "assemble",
]

# Every known stage, so the state machine can resolve any project's stage list by name.
# ReviewStage (automated LLM QC) is no longer wired into any pipeline — clips are reviewed
# by a human at the gate — but it stays resolvable so older projects that still list a
# ``review`` stage can load and resume without error.
STAGE_REGISTRY = STAGES + [ConceptStage(), ClipStage(), ReviewStage()]


def pipeline_for(resolved_format) -> list[str]:
    """The ordered stage names for a resolved format's mode."""
    if format_mode(resolved_format.spec) == "clip":
        return list(CLIP_PIPELINE)
    return list(PIPELINE)


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def new_project_audio_mode(profile: dict) -> str:
    return str(profile.get("audio_mode") or "native_video")


def new_project_music(args, profile: dict) -> dict:
    """Resolve the project's background-music choice. Music is OFF by default; --music/--no-music
    override a profile default. Returns the keys spread into model_config."""
    flag = getattr(args, "music", None)
    if flag is None:
        enabled = bool(profile.get("music_enabled", False))
    else:
        enabled = bool(flag)
    mood = str(getattr(args, "music_mood", None) or "").strip()
    return {"music_enabled": enabled, "music_mood": mood}


def build_providers(profile: dict) -> Providers:
    """Construct provider implementations for a config profile.

    Only the fake LLM is wired for the spine; real/unimplemented providers raise a
    clear error rather than silently doing nothing.
    """
    env_support.load_dotenv()
    providers = Providers()
    if profile.get("llm") == "fake":
        providers.llm = FakeLLM()
        providers.reference_analyzer = FakeReferenceAnalyzer()
        providers.style_profiler = FakeStyleProfiler()
    elif profile.get("llm") in ("openai-compatible", "openai_compatible"):
        from .providers.openai_compatible import OpenAICompatibleLLM
        providers.llm = OpenAICompatibleLLM(
            base_url=profile.get("llm_base_url", "http://localhost:8000/v1"),
            model=profile.get("llm_model", "local-model"),
            api_key_env=profile.get("llm_api_key_env"),
            temperature=float(profile.get("llm_temperature", 0.4)),
            timeout_s=float(profile.get("llm_timeout_s", 120.0)),
            stream=bool(profile.get("llm_stream", True)),
            max_json_attempts=int(profile.get("llm_max_json_attempts", 2)),
            json_response_format=bool(profile.get("llm_json_response_format", True)),
            max_retries=int(profile.get("llm_max_retries", 3)),
            retry_backoff_s=float(profile.get("llm_retry_backoff_s", 1.5)),
            cost_per_million_input=float(
                profile.get("llm_cost_per_million_input", 0.0)
            ),
            cost_per_million_cached_input=float(
                profile.get(
                    "llm_cost_per_million_cached_input",
                    profile.get("llm_cost_per_million_input", 0.0),
                )
            ),
            cost_per_million_output=float(
                profile.get("llm_cost_per_million_output", 0.0)
            ),
            cost_currency=profile.get("llm_cost_currency", "USD"),
            usd_per_native_unit=float(
                profile.get("llm_usd_per_native_unit", 1.0)
            ),
            pricing_source=profile.get("llm_pricing_source", "config"),
            pricing_as_of=profile.get("llm_pricing_as_of", ""),
        )
    elif profile.get("llm") is not None:
        raise NotImplementedError(
            f"LLM provider '{profile['llm']}' not wired yet — use --fake for now."
        )
    if profile.get("image") == "fake":
        providers.image = FakeImageGen(supports_storyboard_grid=bool(profile.get("image_supports_storyboard_grid", True)))
    elif profile.get("image") == "gemini":
        from .providers.gemini import GeminiImageGen, DEFAULT_MODEL as GEMINI_DEFAULT
        providers.image = GeminiImageGen(
            model=profile.get("image_model", GEMINI_DEFAULT),
            max_attempts=int(profile.get("image_max_attempts", 3)),
        )
    elif profile.get("image") == "imagen":
        from .providers.imagen import ImagenImageGen, DEFAULT_MODEL as IMAGEN_DEFAULT
        providers.image = ImagenImageGen(model=profile.get("image_model", IMAGEN_DEFAULT))
    elif profile.get("image") in ("openai", "openai-image", "gpt-image-2", "image-2"):
        from .providers.openai_image import (
            DEFAULT_BASE_URL as OPENAI_IMAGE_BASE_URL,
            DEFAULT_MODEL as OPENAI_IMAGE_DEFAULT,
            OpenAIImageGen,
        )
        providers.image = OpenAIImageGen(
            model=profile.get("image_model", OPENAI_IMAGE_DEFAULT),
            base_url=profile.get("image_base_url", OPENAI_IMAGE_BASE_URL),
            size=profile.get("image_size", "1536x1024"),
            quality=profile.get("image_quality", "medium"),
            api_key_env=profile.get("image_api_key_env", "OPENAI_API_KEY"),
            cost_per_image=float(profile.get("image_cost_per_image", 0.08)),
            max_reference_images=int(profile.get("image_max_reference_images", 10)),
            timeout_s=float(profile.get("image_timeout_s", 120.0)),
        )
    elif profile.get("image") in ("xai", "grok-imagine-image"):
        from .providers.xai_image import (
            DEFAULT_BASE_URL as XAI_IMAGE_BASE_URL,
            DEFAULT_FILE_POLL_INTERVAL_S as XAI_IMAGE_DEFAULT_FILE_POLL_INTERVAL_S,
            DEFAULT_FILE_RECOVERY_TIMEOUT_S as XAI_IMAGE_DEFAULT_FILE_RECOVERY_TIMEOUT_S,
            DEFAULT_MAX_RETRIES as XAI_IMAGE_DEFAULT_MAX_RETRIES,
            DEFAULT_MODEL as XAI_IMAGE_DEFAULT,
            DEFAULT_RETRY_BACKOFF_S as XAI_IMAGE_DEFAULT_RETRY_BACKOFF_S,
            DEFAULT_TIMEOUT_S as XAI_IMAGE_DEFAULT_TIMEOUT_S,
            DEFAULT_USE_FILES_API as XAI_IMAGE_DEFAULT_USE_FILES_API,
            XAIImageGen,
        )
        providers.image = XAIImageGen(
            model=profile.get("image_model", XAI_IMAGE_DEFAULT),
            base_url=profile.get("image_base_url", XAI_IMAGE_BASE_URL),
            api_key_env=profile.get("image_api_key_env", "XAI_API_KEY"),
            aspect_ratio=profile.get("image_aspect_ratio", "16:9"),
            resolution=profile.get("image_resolution", "2k"),
            cost_per_image=float(profile.get("image_cost_per_image", 0.07)),
            cost_per_input_image=float(profile.get("image_cost_per_input_image", 0.01)),
            pricing_source=profile.get(
                "image_pricing_source", "https://docs.x.ai/developers/models"
            ),
            pricing_as_of=profile.get("image_pricing_as_of", ""),
            max_reference_images=int(profile.get("image_max_reference_images", 3)),
            timeout_s=float(profile.get("image_timeout_s", XAI_IMAGE_DEFAULT_TIMEOUT_S)),
            max_retries=int(profile.get("image_max_retries", XAI_IMAGE_DEFAULT_MAX_RETRIES)),
            retry_backoff_s=float(
                profile.get("image_retry_backoff_s", XAI_IMAGE_DEFAULT_RETRY_BACKOFF_S)
            ),
            use_files_api=bool(
                profile.get("xai_use_files_api", XAI_IMAGE_DEFAULT_USE_FILES_API)
            ),
            file_recovery_timeout_s=float(
                profile.get(
                    "xai_file_recovery_timeout_s",
                    XAI_IMAGE_DEFAULT_FILE_RECOVERY_TIMEOUT_S,
                )
            ),
            file_poll_interval_s=float(
                profile.get(
                    "xai_file_poll_interval_s",
                    XAI_IMAGE_DEFAULT_FILE_POLL_INTERVAL_S,
                )
            ),
        )
    elif profile.get("image") in ("doubao", "seedream", "volcengine-image"):
        from .providers.seedream import (
            DEFAULT_BASE_URL as SEEDREAM_BASE_URL,
            DEFAULT_MODEL as SEEDREAM_DEFAULT,
            ArkSeedreamImageGen,
        )
        providers.image = ArkSeedreamImageGen(
            model=profile.get("image_model", SEEDREAM_DEFAULT),
            provider_name=profile.get("image_provider_name", "doubao"),
            provider_label=profile.get("image_provider_label", "Volcengine Ark Seedream"),
            base_url=profile.get("image_base_url", SEEDREAM_BASE_URL),
            api_key_env=profile.get("image_api_key_env", "ARK_API_KEY"),
            size=profile.get("image_size", "2K"),
            aspect_ratio=profile.get("image_aspect_ratio", "16:9"),
            output_format=profile.get("image_output_format"),
            max_reference_images=int(profile.get("image_max_reference_images", 6)),
            poll_interval_s=float(profile.get("image_poll_interval_s", 2.0)),
            max_polls=int(profile.get("image_max_polls", 120)),
            timeout_s=float(profile.get("image_timeout_s", 120.0)),
            cost_per_image_cny=profile.get("image_cost_per_image_cny", 0.2),
            cost_per_image_usd=profile.get("image_cost_per_image_usd"),
            cny_to_usd=float(profile.get("image_cny_to_usd", 0.14)),
        )
    elif profile.get("image") in ("byteplus-seedream", "byteplus-doubao"):
        from .providers.seedream import (
            BYTEPLUS_BASE_URL,
            BYTEPLUS_DEFAULT_COST_PER_IMAGE_USD,
            BYTEPLUS_DEFAULT_MODEL,
            ArkSeedreamImageGen,
        )
        providers.image = ArkSeedreamImageGen(
            model=profile.get("image_model", BYTEPLUS_DEFAULT_MODEL),
            provider_name="byteplus-seedream",
            provider_label="BytePlus ModelArk Seedream",
            base_url=profile.get("image_base_url", BYTEPLUS_BASE_URL),
            api_key_env=profile.get("image_api_key_env", "BYTEPLUS_ARK_API_KEY"),
            size=profile.get("image_size", "2K"),
            aspect_ratio=profile.get("image_aspect_ratio", "16:9"),
            output_format=profile.get("image_output_format"),
            max_reference_images=int(profile.get("image_max_reference_images", 6)),
            poll_interval_s=float(profile.get("image_poll_interval_s", 2.0)),
            max_polls=int(profile.get("image_max_polls", 120)),
            timeout_s=float(profile.get("image_timeout_s", 120.0)),
            cost_per_image_cny=None,
            cost_per_image_usd=float(
                profile.get("image_cost_per_image_usd", BYTEPLUS_DEFAULT_COST_PER_IMAGE_USD)
            ),
        )
    elif profile.get("image") in ("manual-midjourney", "midjourney-manual", "midjourney"):
        from .providers.manual_midjourney import ManualMidjourneyImageGen
        providers.image = ManualMidjourneyImageGen(
            aspect_ratio=profile.get("image_aspect_ratio", "16:9"),
            model=profile.get("image_model", "manual-midjourney-import"),
            cost_per_image=float(profile.get("image_cost_per_image", 0.0)),
        )
    elif profile.get("image") is not None:
        raise NotImplementedError(
            f"image provider '{profile['image']}' not wired yet — use --fake for now."
        )
    if providers.image is not None:
        if "image_max_prompt_length" in profile:
            providers.image.max_prompt_length = int(
                profile["image_max_prompt_length"]
            )
        if "image_prompt_length_unit" in profile:
            providers.image.prompt_length_unit = str(
                profile["image_prompt_length_unit"]
            )
        providers.image._supports_storyboard_grid = bool(
            profile.get("image_supports_storyboard_grid",
                        getattr(providers.image, "_supports_storyboard_grid", False))
        )
        # Constructing capabilities validates configured prompt limits before any paid call.
        _ = providers.image.capabilities
    if profile.get("video") == "fake":
        providers.video = FakeVideoGen(supports_storyboard_grid=bool(profile.get("video_supports_storyboard_grid", True)))
    elif profile.get("video") == "fal":
        from .providers.fal import FalReferenceVideoGen, DEFAULT_MODEL as FAL_DEFAULT
        providers.video = FalReferenceVideoGen(
            model=profile.get("video_model", FAL_DEFAULT),
            resolution=profile.get("video_resolution", "720p"),
            aspect_ratio=profile.get("video_aspect_ratio", "16:9"),
            generate_audio=bool(profile.get("video_generate_audio", False)),
        )
    elif profile.get("video") in ("gemini-veo", "veo", "gemini-video"):
        from .providers.gemini_video import DEFAULT_MODEL as VEO_DEFAULT, GeminiVeoVideoGen
        providers.video = GeminiVeoVideoGen(
            model=profile.get("video_model", VEO_DEFAULT),
            api_key_env=profile.get("video_api_key_env", "GEMINI_API_KEY"),
            resolution=profile.get("video_resolution", "720p"),
            aspect_ratio=profile.get("video_aspect_ratio", "16:9"),
            generate_audio=bool(profile.get("video_generate_audio", True)),
            cost_per_second=float(profile.get("video_cost_per_second", 0.0)),
            poll_interval_s=float(profile.get("video_poll_interval_s", 10.0)),
            max_polls=int(profile.get("video_max_polls", 180)),
            supports_reference_images=bool(profile.get("video_include_reference_images", True)),
            supports_last_frame=bool(profile.get("video_include_last_frame_ref", True)),
            max_image_inputs=int(profile.get("video_max_image_inputs", 4)),
        )
    elif profile.get("video") in ("xai", "grok-imagine-video"):
        from .providers.xai_video import (
            DEFAULT_BASE_URL as XAI_VIDEO_BASE_URL,
            DEFAULT_MODEL as XAI_VIDEO_DEFAULT,
            XAIVideoGen,
        )
        providers.video = XAIVideoGen(
            model=profile.get("video_model", XAI_VIDEO_DEFAULT),
            base_url=profile.get("video_base_url", XAI_VIDEO_BASE_URL),
            api_key_env=profile.get("video_api_key_env", "XAI_API_KEY"),
            resolution=profile.get("video_resolution", "720p"),
            aspect_ratio=profile.get("video_aspect_ratio", "16:9"),
            cost_per_second=float(profile.get("video_cost_per_second", 0.07)),
            cost_per_input_image=float(profile.get("video_cost_per_input_image", 0.002)),
            poll_interval_s=float(profile.get("video_poll_interval_s", 5.0)),
            max_polls=int(profile.get("video_max_polls", 180)),
            timeout_s=float(profile.get("video_timeout_s", 60.0)),
        )
    elif profile.get("video") == "volcengine":
        from .providers.volcengine import (
            DEFAULT_BASE_URL as ARK_BASE_URL,
            DEFAULT_COST_PER_MILLION_TOKENS_CNY,
            DEFAULT_MODEL as ARK_DEFAULT,
            ArkSeedanceVideoGen,
        )
        providers.video = ArkSeedanceVideoGen(
            model=profile.get("video_model", ARK_DEFAULT),
            base_url=profile.get("video_base_url", ARK_BASE_URL),
            resolution=profile.get("video_resolution", "720p"),
            aspect_ratio=profile.get("video_aspect_ratio", "16:9"),
            generate_audio=bool(profile.get("video_generate_audio", False)),
            return_last_frame=bool(profile.get("video_return_last_frame", True)),
            api_key_env=profile.get("video_api_key_env", "ARK_API_KEY"),
            cost_per_million_tokens_cny=profile.get(
                "video_cost_per_million_tokens_cny",
                DEFAULT_COST_PER_MILLION_TOKENS_CNY,
            ),
            cny_to_usd=float(profile.get("video_cny_to_usd", 0.14)),
            cost_per_1k_tokens_usd=profile.get("video_cost_per_1k_tokens_usd"),
            pricing_source=profile.get("video_pricing_source", "config"),
            pricing_as_of=profile.get("video_pricing_as_of", ""),
            poll_interval_s=float(profile.get("video_poll_interval_s", 5.0)),
            max_polls=int(profile.get("video_max_polls", 180)),
            timeout_s=float(profile.get("video_timeout_s", 60.0)),
            include_reference_images=bool(profile.get("video_include_reference_images", True)),
            include_last_frame_ref=bool(profile.get("video_include_last_frame_ref", True)),
        )
    elif profile.get("video") == "byteplus":
        from .providers.volcengine import (
            BYTEPLUS_BASE_URL,
            BYTEPLUS_DEFAULT_COST_PER_1K_TOKENS_USD,
            BYTEPLUS_DEFAULT_MODEL,
            ArkSeedanceVideoGen,
        )
        providers.video = ArkSeedanceVideoGen(
            model=profile.get("video_model", BYTEPLUS_DEFAULT_MODEL),
            provider_name="byteplus",
            provider_label="BytePlus ModelArk",
            base_url=profile.get("video_base_url", BYTEPLUS_BASE_URL),
            resolution=profile.get("video_resolution", "720p"),
            aspect_ratio=profile.get("video_aspect_ratio", "16:9"),
            generate_audio=bool(profile.get("video_generate_audio", False)),
            return_last_frame=bool(profile.get("video_return_last_frame", True)),
            api_key_env=profile.get("video_api_key_env", "BYTEPLUS_ARK_API_KEY"),
            cost_per_million_tokens_cny=None,
            cny_to_usd=float(profile.get("video_cny_to_usd", 0.14)),
            cost_per_1k_tokens_usd=profile.get(
                "video_cost_per_1k_tokens_usd",
                BYTEPLUS_DEFAULT_COST_PER_1K_TOKENS_USD,
            ),
            pricing_source=profile.get("video_pricing_source", "config"),
            pricing_as_of=profile.get("video_pricing_as_of", ""),
            poll_interval_s=float(profile.get("video_poll_interval_s", 5.0)),
            max_polls=int(profile.get("video_max_polls", 180)),
            timeout_s=float(profile.get("video_timeout_s", 60.0)),
            include_reference_images=bool(profile.get("video_include_reference_images", False)),
            include_last_frame_ref=bool(profile.get("video_include_last_frame_ref", False)),
        )
    elif profile.get("video") is not None:
        raise NotImplementedError(
            f"video provider '{profile['video']}' not wired yet — use --fake for now."
        )
    if providers.video is not None:
        providers.video._supports_storyboard_grid = bool(
            profile.get("video_supports_storyboard_grid",
                        getattr(providers.video, "_supports_storyboard_grid", False))
        )
    if profile.get("vlm") == "fake":
        providers.vlm = FakeVLMCheck()
    elif profile.get("vlm") == "anthropic":
        from .providers.anthropic_vlm import AnthropicVLMCheck, DEFAULT_MODEL as VLM_DEFAULT
        providers.vlm = AnthropicVLMCheck(
            model=profile.get("vlm_model", VLM_DEFAULT),
            max_frames=int(profile.get("vlm_max_frames", 4)),
        )
    elif profile.get("vlm") == "gemini":
        from .providers.gemini_vlm import GeminiVLMCheck, DEFAULT_MODEL as VLM_DEFAULT
        providers.vlm = GeminiVLMCheck(
            model=profile.get("vlm_model", VLM_DEFAULT),
            max_tokens=int(profile.get("vlm_max_tokens", 2048)),
            media_resolution=profile.get("vlm_media_resolution", "low"),
            inline_video_max_mb=float(profile.get("vlm_inline_video_max_mb", 20.0)),
            max_attempts=int(profile.get("vlm_max_attempts", 3)),
        )
    elif profile.get("vlm") in (
        "openai-compatible-vision",
        "openai_compatible_vision",
        "ark-vlm",
        "doubao-vlm",
        "byteplus-vlm",
    ):
        from .providers.openai_compatible_vlm import (
            DEFAULT_BASE_URL as VLM_BASE_URL,
            DEFAULT_MODEL as VLM_DEFAULT,
            OpenAICompatibleVLMCheck,
        )
        providers.vlm = OpenAICompatibleVLMCheck(
            base_url=profile.get("vlm_base_url", VLM_BASE_URL),
            model=profile.get("vlm_model", VLM_DEFAULT),
            provider_name=profile.get("vlm_provider_name", profile.get("vlm", "ark-vlm")),
            api_key_env=profile.get("vlm_api_key_env"),
            max_frames=int(profile.get("vlm_max_frames", 4)),
            max_tokens=int(profile.get("vlm_max_tokens", 2048)),
            detail=profile.get("vlm_detail"),
            temperature=float(profile.get("vlm_temperature", 0.0)),
            timeout_s=float(profile.get("vlm_timeout_s", 120.0)),
            json_response_format=bool(profile.get("vlm_json_response_format", False)),
            cost_per_1k_input_usd=float(profile.get("vlm_cost_per_1k_input_usd", 0.0)),
            cost_per_1k_output_usd=float(profile.get("vlm_cost_per_1k_output_usd", 0.0)),
            cost_per_million_input=(
                float(profile["vlm_cost_per_million_input"])
                if "vlm_cost_per_million_input" in profile
                else None
            ),
            cost_per_million_cached_input=(
                float(profile["vlm_cost_per_million_cached_input"])
                if "vlm_cost_per_million_cached_input" in profile
                else None
            ),
            cost_per_million_output=(
                float(profile["vlm_cost_per_million_output"])
                if "vlm_cost_per_million_output" in profile
                else None
            ),
            cost_currency=profile.get("vlm_cost_currency", "USD"),
            usd_per_native_unit=float(
                profile.get("vlm_usd_per_native_unit", 1.0)
            ),
            pricing_source=profile.get("vlm_pricing_source", "config"),
            pricing_as_of=profile.get("vlm_pricing_as_of", ""),
        )
    elif profile.get("vlm") is not None:
        raise NotImplementedError(
            f"VLM provider '{profile['vlm']}' not wired yet — use --fake for now."
        )
    # Any vision-capable VLM doubles as the reference analyzer, so uploaded images ground the
    # bible no matter which provider is selected (overriding the fake default from --fake).
    if isinstance(providers.vlm, ReferenceAnalyzer):
        providers.reference_analyzer = providers.vlm
    if profile.get("tts") == "fake":
        providers.tts = FakeTTS()
    elif profile.get("tts") is not None:
        raise NotImplementedError(
            f"TTS provider '{profile['tts']}' not wired yet — use --fake for now."
        )
    if profile.get("music") == "fake":
        providers.music = FakeMusic()
    elif profile.get("music") is not None:
        raise NotImplementedError(
            f"music provider '{profile['music']}' not wired yet — use --fake for now."
        )
    _wire_style_profiler(providers, profile)
    return providers


def _wire_style_profiler(providers: Providers, profile: dict) -> None:
    """Attach a style profiler (free text / image -> style dict) for custom styles."""
    kind = profile.get("style_profiler")
    if kind == "fake":
        providers.style_profiler = FakeStyleProfiler()
    elif kind in (
        "openai-compatible", "openai_compatible", "openai-compatible-vision",
        "ark-vlm", "doubao-vlm", "byteplus-vlm",
    ):
        from .providers.openai_compatible_vlm import (
            DEFAULT_BASE_URL as STYLE_BASE_URL,
            DEFAULT_MODEL as STYLE_DEFAULT,
            OpenAICompatibleStyleProfiler,
        )
        providers.style_profiler = OpenAICompatibleStyleProfiler(
            base_url=profile.get("style_profiler_base_url", profile.get("vlm_base_url", STYLE_BASE_URL)),
            model=profile.get("style_profiler_model", profile.get("vlm_model", STYLE_DEFAULT)),
            provider_name=profile.get("style_profiler_provider_name", "style-profiler"),
            api_key_env=profile.get("style_profiler_api_key_env", profile.get("vlm_api_key_env")),
            detail=profile.get("style_profiler_detail", profile.get("vlm_detail")),
            timeout_s=float(profile.get("style_profiler_timeout_s", 120.0)),
            json_response_format=bool(profile.get("style_profiler_json_response_format", False)),
            cost_per_1k_input_usd=float(profile.get("style_profiler_cost_per_1k_input_usd", 0.0)),
            cost_per_1k_output_usd=float(profile.get("style_profiler_cost_per_1k_output_usd", 0.0)),
        )
    elif kind is not None:
        raise NotImplementedError(
            f"style profiler '{kind}' not wired yet — use --fake for now."
        )
    elif providers.style_profiler is None:
        # No explicit profiler: prefer the vision model the profile already configures for
        # reference analysis (vlm_*), so an uploaded style image is actually *seen*. Only
        # the OpenAI-compatible vision family implements style profiling today; for a
        # gemini/anthropic/absent vlm, fall back to the text-only LLM profiler.
        vlm_kind = profile.get("vlm")
        if vlm_kind in (
            "openai-compatible-vision", "openai_compatible_vision",
            "ark-vlm", "doubao-vlm", "byteplus-vlm",
        ):
            from .providers.openai_compatible_vlm import (
                DEFAULT_BASE_URL as STYLE_BASE_URL,
                DEFAULT_MODEL as STYLE_DEFAULT,
                OpenAICompatibleStyleProfiler,
            )
            providers.style_profiler = OpenAICompatibleStyleProfiler(
                base_url=profile.get("vlm_base_url", STYLE_BASE_URL),
                model=profile.get("vlm_model", STYLE_DEFAULT),
                provider_name="style-profiler",
                api_key_env=profile.get("vlm_api_key_env"),
                detail=profile.get("vlm_detail"),
                timeout_s=float(profile.get("vlm_timeout_s", 120.0)),
                json_response_format=bool(profile.get("vlm_json_response_format", False)),
                cost_per_1k_input_usd=float(profile.get("vlm_cost_per_1k_input_usd", 0.0)),
                cost_per_1k_output_usd=float(profile.get("vlm_cost_per_1k_output_usd", 0.0)),
            )
        elif vlm_kind == "gemini":
            from .providers.gemini_vlm import DEFAULT_STYLE_MODEL, GeminiStyleProfiler
            providers.style_profiler = GeminiStyleProfiler(
                model=profile.get("vlm_model", DEFAULT_STYLE_MODEL),
                api_key_env=profile.get("vlm_api_key_env", "GEMINI_API_KEY"),
                cost_per_1k_input_usd=float(profile.get("vlm_cost_per_1k_input_usd", 0.0001)),
                cost_per_1k_output_usd=float(profile.get("vlm_cost_per_1k_output_usd", 0.0004)),
            )
        elif vlm_kind == "anthropic":
            from .providers.anthropic_vlm import AnthropicStyleProfiler, DEFAULT_STYLE_MODEL
            providers.style_profiler = AnthropicStyleProfiler(
                model=profile.get("vlm_model", DEFAULT_STYLE_MODEL),
                api_key_env=profile.get("vlm_api_key_env", "ANTHROPIC_API_KEY"),
                cost_per_1k_input_usd=float(profile.get("vlm_cost_per_1k_input_usd", 0.005)),
                cost_per_1k_output_usd=float(profile.get("vlm_cost_per_1k_output_usd", 0.025)),
            )
        elif providers.llm is not None:
            from .providers.llm_style import LLMStyleProfiler
            providers.style_profiler = LLMStyleProfiler(providers.llm)


def _machine() -> StateMachine:
    return StateMachine(STAGE_REGISTRY)


def _resolve_dir(project_id: str) -> Path:
    return PROJECTS_ROOT / project_id


def cmd_run(args) -> int:
    config = load_config()
    if args.profile:
        profile_name = args.profile
    elif args.fake:
        profile_name = "fake"
    else:
        profile_name = config.get("default_profile", "fake")
    if profile_name not in config["profiles"]:
        print(f"unknown profile '{profile_name}'. Available: {', '.join(config['profiles'])}")
        return 1
    profile = config["profiles"][profile_name]
    style_description = (getattr(args, "style_description", "") or "").strip()
    style_image = (getattr(args, "style_image", "") or "").strip()
    custom_style = bool(style_description or style_image)
    if style_image and not Path(style_image).is_file():
        print(f"style image not found: {style_image}")
        return 1
    try:
        resolved_style = resolve_style(
            config, CUSTOM_STYLE_NAME if custom_style else args.style
        )
    except UnknownStyleError as exc:
        print(f"unknown style '{exc.name}'. Available: {', '.join(exc.available)}")
        return 1
    try:
        resolved_format = resolve_format(config, args.format)
    except UnknownFormatError as exc:
        print(f"unknown format '{exc.name}'. Available: {', '.join(exc.available)}")
        return 1
    try:
        language = resolve_language(args.idea, args.language)
    except UnknownLanguageError as exc:
        print("unknown language "
              f"'{exc.name}'. Available: auto, en, zh")
        return 1

    creative_inputs = {
        key: value
        for key in (
            "tone",
            "visual_world",
            "camera_language",
            "light_texture",
            "character_treatment",
        )
        if (value := str(getattr(args, key, "") or "").strip())
    }
    hard_avoidances = list(getattr(args, "hard_avoidance", None) or [])
    if hard_avoidances:
        creative_inputs["hard_avoidances"] = hard_avoidances

    mode = format_mode(resolved_format.spec)
    length = args.clip_duration if mode == "clip" else args.duration_min
    sized_spec = apply_length(resolved_format.spec, mode, length)
    clip_runtime = {}
    if mode == "clip":
        clip_runtime["clip_target_duration_s"] = sized_spec["clip_target_duration_s"]
        if args.clips is not None or args.clip_seconds is not None:
            clip_runtime["clip_plan_mode"] = "manual"
        elif args.clip_duration == "auto":
            clip_runtime["clip_plan_mode"] = "auto"
        elif args.clip_duration is not None:
            pass  # legacy fixed total runtime — the numeric target drives the split
        else:
            clip_runtime["clip_plan_mode"] = "default"

    providers = build_providers(profile)
    if mode == "clip" and args.clip_seconds is not None:
        capabilities = getattr(getattr(providers, "video", None), "capabilities", None)
        if capabilities is not None and not (
            capabilities.min_duration_s <= args.clip_seconds <= capabilities.max_duration_s
        ):
            print(
                f"--clip-seconds {args.clip_seconds} is outside the video model's supported "
                f"range [{capabilities.min_duration_s}, {capabilities.max_duration_s}]"
            )
            return 1

    project = Project.create(
        args.idea,
        root=PROJECTS_ROOT,
        stages=pipeline_for(resolved_format),
        cost_cap=config.get("cost_cap"),
        model_config={
            "profile": profile_name,
            "style_name": resolved_style.name,
            "style": resolved_style.style,
            "genre": (args.genre or "").strip(),
            "format_name": resolved_format.name,
            "product_format": {"name": resolved_format.name, **sized_spec},
            "language": language,
            "clip_count": args.clips or resolved_format.spec.get("clip_count", 1),
            "clip_seconds": args.clip_seconds or resolved_format.spec.get("clip_seconds", 15),
            **clip_runtime,
            **profile,
            "audio_mode": new_project_audio_mode(profile),
            **new_project_music(args, profile),
            "motion_grid": config.get("motion_grid", {"enabled": False, "layout": "auto"}),
            "storyboard": config.get("storyboard", {"flow_review": {"enabled": True}}),
            "creative_inputs": creative_inputs,
            **({"style_input": {"description": style_description}} if custom_style else {}),
        },
    )
    project.story_dir.joinpath("idea.md").write_text(args.idea + "\n")
    if style_image:
        image_path = Path(style_image)
        save_reference_upload(
            project,
            target_type="style",
            target_id="global",
            data=image_path.read_bytes(),
            filename=image_path.name,
            label="style reference",
        )
    project.save()

    result = _machine().run(project, providers, auto=args.auto)
    _report(project, result)
    return 0


def cmd_resume(args) -> int:
    project = Project.load(_resolve_dir(args.project_id))
    profile = project.model_config or load_config()["profiles"]["fake"]
    providers = build_providers(profile)
    result = _machine().run(project, providers, auto=args.auto)
    _report(project, result)
    return 0


def cmd_approve(args) -> int:
    project = Project.load(_resolve_dir(args.project_id))
    new_stage = _machine().approve(project)
    if new_stage is None:
        print(f"[{project.project_id}] all stages approved — project done.")
    else:
        print(f"[{project.project_id}] approved; next stage: {new_stage}")
    return 0


def cmd_answer_decision(args) -> int:
    project = Project.load(_resolve_dir(args.project_id))
    data = resolve_decision(project, stage=args.stage, choice=args.choice)
    print(
        f"[{project.project_id}] resolved {args.stage}: "
        f"{data['resolution']['value']} (user)"
    )
    return 0


def cmd_knowledge_import(args) -> int:
    source = Path(args.path)
    entries = import_knowledge_file(
        source,
        root=KNOWLEDGE_ROOT,
        metadata={
            "domain": args.domain or "general",
            "stages": list(args.stage or []),
            "language": args.language or "und",
        },
    )
    index_path = rebuild_index(KNOWLEDGE_ROOT)
    manifest_path = KNOWLEDGE_ROOT / "imports" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    record = next(
        (
            item for item in manifest.get("sources", [])
            if Path(str(item.get("original_path") or "")) == source
        ),
        {},
    )
    print(f"source: {record.get('id') or source.stem}")
    print(f"chunks: {len(entries)}")
    print(f"manifest: {manifest_path}")
    print(f"index: {index_path}")
    return 0


def cmd_knowledge_rebuild_index(args) -> int:
    path = rebuild_index(KNOWLEDGE_ROOT)
    print(f"index: {path}")
    return 0


def cmd_knowledge_refresh(args) -> int:
    project = Project.load(_resolve_dir(args.project_id))
    result = refresh_knowledge_target(
        project,
        purpose=args.purpose,
        target=args.target,
    )
    print(
        f"[{project.project_id}] refreshed {args.purpose}:{args.target}; "
        f"archived {result.archived} artifact(s); "
        f"downstream pending: {', '.join(result.invalidated_stages) or 'none'}"
    )
    return 0


def cmd_regenerate(args) -> int:
    project = Project.load(_resolve_dir(args.project_id))
    shot = _find_shot(project, args.shot)
    if shot is None:
        print(f"[{project.project_id}] shot not found: {args.shot}")
        return 1
    if "video" not in project.stages:
        print(f"[{project.project_id}] shot regeneration requires the video stage.")
        return 1

    regeneration = regenerate_shot_video(project, args.shot)
    removed = regeneration.removed

    profile = project.model_config or load_config()["profiles"]["fake"]
    providers = build_providers(profile)
    result = _machine().run(project, providers, auto=True)
    _report(project, result)
    print(f"[{project.project_id}] regenerated shot {args.shot}; "
          f"refreshed {removed} artifact(s).")
    return 0 if result.done else 1


def cmd_doctor(args) -> int:
    env_path = env_support.DEFAULT_ENV_PATH
    env_support.load_dotenv(env_path)
    print("Studio Agent doctor")
    print(f"env file: {env_path} ({'found' if env_path.is_file() else 'missing'})")
    print("api keys:")
    for key, is_set in env_support.api_key_status().items():
        print(f"  {key}: {'set' if is_set else 'missing'}")
    ffmpeg = shutil.which("ffmpeg")
    print(f"ffmpeg: {ffmpeg or 'missing'}")
    return 0


def cmd_web(args) -> int:
    from .web import run_server
    run_server(PROJECTS_ROOT, host=args.host, port=args.port)
    return 0


def _find_shot(project: Project, shot_id: str) -> dict | None:
    shots_path = project.path("storyboard", "shots.json")
    if not shots_path.is_file():
        return None
    for shot in json.loads(shots_path.read_text()).get("shots", []):
        if shot.get("id") == shot_id:
            return shot
    return None


def _report(project: Project, result) -> None:
    print(f"project: {project.project_id}  ({project.dir})")
    print(f"cost so far: ${project.total_cost():.2f} / cap ${project.cost_cap}")
    if result.done:
        print("pipeline complete — all stages approved.")
    elif getattr(result, "manual_import_required", False):
        print(
            f"manual import required at stage '{result.paused_at}'. "
            f"Follow `{result.request_path}`, save the image to the requested path, "
            "then resume."
        )
    elif getattr(result, "cost_capped", False):
        print(
            f"cost cap reached at stage '{result.paused_at}'. "
            "Inspect artifacts, raise the cap in config/project settings, then resume."
        )
    else:
        print(f"paused at human gate: '{result.paused_at}'. "
              f"Edit its files, then `approve {project.project_id}` to continue.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="studio_agent", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="start a new project from an idea")
    run.add_argument("--idea", required=True, help="one-line idea")
    run.add_argument("--fake", action="store_true", help="use offline zero-cost providers")
    run.add_argument("--profile", help="config profile to use (e.g. real-image); overrides --fake")
    run.add_argument("--style", help="named style preset from config (e.g. anime, cartoon)")
    run.add_argument(
        "--style-description",
        help="define a custom project-wide style in your own words (e.g. '1970s grainy "
        "analog sci-fi'); the style stage profiles it and locks it for the whole film",
    )
    run.add_argument(
        "--style-image",
        help="path to a reference image whose style only is extracted and locked for the "
        "whole film (its subject/composition is never copied)",
    )
    run.add_argument(
        "--genre",
        help="lock a genre/题材 (e.g. 中国古代神话, 东方奇幻) that cascades into story and "
        "picture prompts; omit to infer from the idea",
    )
    run.add_argument(
        "--format",
        help="product format preset (e.g. short_film, short_video)",
    )
    run.add_argument(
        "--duration-min",
        dest="duration_min",
        default=None,
        help="film/story mode only: target runtime in minutes (default from format)",
    )
    run.add_argument(
        "--language",
        default="auto",
        help="project language: auto, en, or zh",
    )
    run.add_argument(
        "--clips", type=int, default=None,
        help="clip mode: exact number of clips/keyframes (default 1)",
    )
    run.add_argument(
        "--clip-seconds", dest="clip_seconds", type=int, default=None,
        help="clip mode: seconds per clip, up to the video model's max; "
        "omit to use the model's default clip length",
    )
    run.add_argument(
        "--clip-duration",
        choices=("auto", "15", "30", "60"),
        default=None,
        help="clip mode: 'auto' lets the LLM plan the clip count/split from the idea; "
        "a number fixes the total runtime in seconds. Omit (with no --clips) for the "
        "default single clip at the video model's default length",
    )
    run.add_argument(
        "--music", action=argparse.BooleanOptionalAction, default=None,
        help="add a background music score (default off); --no-music to force off",
    )
    run.add_argument("--music-mood", dest="music_mood", default=None,
                     help="optional mood/style steer for the score")
    run.add_argument("--auto", action="store_true", help="skip human gates (hands-off)")
    run.add_argument("--tone", help="emotional tone to lock at kickoff")
    run.add_argument("--visual-world", dest="visual_world", help="visual world to lock")
    run.add_argument("--camera-language", dest="camera_language", help="camera behavior to lock")
    run.add_argument("--light-texture", dest="light_texture", help="lighting texture to lock")
    run.add_argument(
        "--character-treatment",
        dest="character_treatment",
        help="character rendering/performance treatment to lock",
    )
    run.add_argument(
        "--hard-avoidance",
        action="append",
        default=[],
        help="hard prompt avoidance; repeat for multiple rules",
    )
    run.set_defaults(func=cmd_run)

    resume = sub.add_parser("resume", help="continue an interrupted project")
    resume.add_argument("project_id")
    resume.add_argument("--auto", action="store_true")
    resume.set_defaults(func=cmd_resume)

    approve = sub.add_parser("approve", help="pass the current human gate")
    approve.add_argument("project_id")
    approve.set_defaults(func=cmd_approve)

    answer = sub.add_parser("answer-decision", help="resolve a pending creative decision")
    answer.add_argument("project_id")
    answer.add_argument("--stage", required=True)
    answer.add_argument("--choice", required=True)
    answer.set_defaults(func=cmd_answer_decision)

    regen = sub.add_parser("regenerate", help="regenerate one shot")
    regen.add_argument("project_id")
    regen.add_argument("--shot", required=True)
    regen.set_defaults(func=cmd_regenerate)

    doctor = sub.add_parser("doctor", help="check local setup without printing secrets")
    doctor.set_defaults(func=cmd_doctor)

    web = sub.add_parser("web", help="start the local Studio Agent web app")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8765)
    web.set_defaults(func=cmd_web)

    knowledge = sub.add_parser("knowledge", help="manage the local filmmaking corpus")
    knowledge_sub = knowledge.add_subparsers(dest="knowledge_command", required=True)
    knowledge_import = knowledge_sub.add_parser("import", help="import Markdown or text")
    knowledge_import.add_argument("path")
    knowledge_import.add_argument("--domain", default="general")
    knowledge_import.add_argument("--stage", action="append", default=[])
    knowledge_import.add_argument("--language", choices=("en", "zh", "und"), default="und")
    knowledge_import.set_defaults(func=cmd_knowledge_import)
    knowledge_rebuild = knowledge_sub.add_parser("rebuild-index", help="rebuild derived index")
    knowledge_rebuild.set_defaults(func=cmd_knowledge_rebuild_index)
    knowledge_refresh = knowledge_sub.add_parser(
        "refresh", help="refresh one frozen project knowledge packet"
    )
    knowledge_refresh.add_argument("project_id")
    knowledge_refresh.add_argument(
        "--purpose", required=True, choices=("identity", "scene", "shot")
    )
    knowledge_refresh.add_argument("--target", required=True)
    knowledge_refresh.set_defaults(func=cmd_knowledge_refresh)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
