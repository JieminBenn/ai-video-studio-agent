"""Tests for product format and language helpers."""

import pytest

from studio_agent.cli import load_config
from studio_agent.formats import (
    UnknownFormatError,
    apply_length,
    format_mode,
    genre_persona,
    genre_prompt,
    resolve_format,
)
from studio_agent.language import UnknownLanguageError, resolve_language


def test_genre_prompt_emits_a_locked_genre_line_when_set():
    out = genre_prompt({"genre": "中国古代神话"})
    assert "中国古代神话" in out
    assert "GENRE" in out
    assert out.endswith("\n")


def test_genre_prompt_is_empty_when_genre_absent_or_blank():
    assert genre_prompt({}) == ""
    assert genre_prompt({"genre": "   "}) == ""
    assert genre_prompt(None) == ""


def test_genre_persona_assigns_a_genre_expert_identity_when_set():
    # The tutorial's 是什么 step: give the model a genre-specific expert identity.
    out = genre_persona({"genre": "中国古代神话"})
    assert "中国古代神话" in out
    # It slots before a role noun (e.g. "You are {persona}a showrunner"), so it must
    # end with a separator space and read as a leading qualifier.
    assert out.endswith(" ")


def test_genre_persona_is_empty_when_genre_absent_or_blank():
    assert genre_persona({}) == ""
    assert genre_persona({"genre": "   "}) == ""
    assert genre_persona(None) == ""


CONFIG = {
    "default_format": "short_film",
    "product_formats": {
        "short_film": {"target_duration_s": 180},
        "short_drama_episode": {"target_duration_s": 75, "min_duration_s": 60, "max_duration_s": 90},
    },
}


def test_resolve_default_format():
    resolved = resolve_format(CONFIG)

    assert resolved.name == "short_film"
    assert resolved.to_project_config()["target_duration_s"] == 180


def test_resolve_unknown_format_lists_available():
    with pytest.raises(UnknownFormatError) as exc:
        resolve_format(CONFIG, "feature_film")

    assert exc.value.name == "feature_film"
    assert exc.value.available == ["short_drama_episode", "short_film"]


def test_language_auto_detects_chinese_and_english():
    assert resolve_language("a lonely lighthouse keeper") == "en"
    assert resolve_language("一个灯塔守望者遇见会说话的海鸥") == "zh"


def test_unknown_language_fails():
    with pytest.raises(UnknownLanguageError):
        resolve_language("idea", "klingon")


def test_short_video_resolves_to_clip_mode():
    config = load_config()
    resolved = resolve_format(config, "short_video")
    assert resolved.spec["mode"] == "clip"
    assert resolved.spec["clip_count"] == 1
    # null = the video model's own default clip length (no fixed seconds).
    assert resolved.spec["clip_seconds"] is None


def test_existing_format_defaults_to_story_mode():
    config = load_config()
    resolved = resolve_format(config, "short_film")
    assert format_mode(resolved.spec) == "story"


def test_format_mode_reads_clip():
    assert format_mode({"mode": "clip"}) == "clip"
    assert format_mode({}) == "story"
    assert format_mode(None) == "story"


def test_apply_length_clip_sets_target_seconds():
    spec = {"mode": "clip", "clip_target_duration_s": "auto"}
    out = apply_length(spec, "clip", "30")
    assert out["clip_target_duration_s"] == 30


def test_apply_length_clip_auto_keeps_auto():
    spec = {"mode": "clip", "clip_target_duration_s": "auto"}
    out = apply_length(spec, "clip", "auto")
    assert out["clip_target_duration_s"] == "auto"


def test_apply_length_story_minutes_override_duration_bounds():
    spec = {"target_duration_s": 180, "min_duration_s": 120, "max_duration_s": 300}
    out = apply_length(spec, "story", "2")
    # 2 minutes -> 120s target, with proportional bounds.
    assert out["target_duration_s"] == 120
    assert out["min_duration_s"] == 84
    assert out["max_duration_s"] == 168


def test_apply_length_story_auto_preserves_preset_bounds():
    spec = {"target_duration_s": 180, "min_duration_s": 120, "max_duration_s": 300}
    out = apply_length(spec, "story", "auto")
    assert out["target_duration_s"] == 180
    assert out["min_duration_s"] == 120
    assert out["max_duration_s"] == 300


def test_apply_length_does_not_mutate_input_spec():
    spec = {"target_duration_s": 180, "min_duration_s": 120, "max_duration_s": 300}
    apply_length(spec, "story", "5")
    assert spec["target_duration_s"] == 180


def test_motion_grid_defaults_present_and_disabled():
    cfg = load_config()
    mg = cfg.get("motion_grid") or {}
    assert mg.get("enabled") is False
    assert mg.get("layout") == "auto"
