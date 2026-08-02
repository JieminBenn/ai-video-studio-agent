"""Tests for the cinematography recipe library (deterministic, offline)."""

from __future__ import annotations

import textwrap

import pytest

from studio_agent.cinematography_recipes import (
    fill_recipe_slots,
    load_camera_recipes,
    recipe_menu,
    resolve_recipe,
)


def _write_recipes(tmp_path, body: str):
    path = tmp_path / "recipes.yaml"
    path.write_text(textwrap.dedent(body))
    return path


SAMPLE = """
    - id: orbit_push_360
      family: orbit
      name_zh: 360环绕+急推
      name_en: 360° orbit into rapid push-in
      intents_en: [villain reveal, character turn dark, emotional peak]
      intents_zh: [反派登场, 角色黑化, 情绪爆发顶点]
      motion_strength: high
      prompt_zh: "360°环绕运镜，极推到主体眼睛特写"
      prompt_en: "360° orbital camera, rapid push-in to a close-up of the subject's eyes"
    - id: pan_crane_reveal
      family: crane
      name_zh: 摇镜+升镜
      name_en: pan into crane-up reveal
      intents_en: [lonely grandeur, vast environment, sense of fate]
      intents_zh: [崇高, 孤独, 宿命感]
      motion_strength: medium
      prompt_zh: "水平摇镜随即升镜到高空上帝视角全景"
      prompt_en: "horizontal pan then crane up to a high god's-eye wide of the environment"
"""


def test_load_camera_recipes_keys_by_id(tmp_path):
    path = _write_recipes(tmp_path, SAMPLE)
    recipes = load_camera_recipes(path=path)
    assert set(recipes) == {"orbit_push_360", "pan_crane_reveal"}
    assert recipes["orbit_push_360"]["family"] == "orbit"


def test_load_missing_file_returns_empty(tmp_path):
    assert load_camera_recipes(path=tmp_path / "nope.yaml") == {}


def test_packaged_recipe_file_loads_and_is_nonempty():
    """The shipped seed library must parse and carry several recipes."""
    recipes = load_camera_recipes()
    assert len(recipes) >= 5
    for recipe in recipes.values():
        assert recipe.get("id")
        assert recipe.get("prompt_en") and recipe.get("prompt_zh")
        assert recipe.get("intents_en") and recipe.get("intents_zh")


def test_recipe_menu_english(tmp_path):
    recipes = load_camera_recipes(path=_write_recipes(tmp_path, SAMPLE))
    menu = recipe_menu(recipes, language="en")
    assert "orbit_push_360" in menu
    assert "360° orbit into rapid push-in" in menu
    assert "villain reveal" in menu
    # English menu must not leak the Chinese name.
    assert "360环绕" not in menu


def test_recipe_menu_chinese(tmp_path):
    recipes = load_camera_recipes(path=_write_recipes(tmp_path, SAMPLE))
    menu = recipe_menu(recipes, language="zh")
    assert "orbit_push_360" in menu
    assert "360环绕+急推" in menu
    assert "反派登场" in menu


def test_resolve_recipe_picks_language(tmp_path):
    recipes = load_camera_recipes(path=_write_recipes(tmp_path, SAMPLE))

    en = resolve_recipe(recipes, "orbit_push_360", language="en")
    assert en["name"] == "360° orbit into rapid push-in"
    assert en["prompt"].startswith("360° orbital camera")
    assert en["intents"] == ["villain reveal", "character turn dark", "emotional peak"]

    zh = resolve_recipe(recipes, "orbit_push_360", language="zh")
    assert zh["name"] == "360环绕+急推"
    assert "极推" in zh["prompt"]
    assert zh["intents"] == ["反派登场", "角色黑化", "情绪爆发顶点"]


def test_resolve_recipe_unknown_id_returns_none(tmp_path):
    recipes = load_camera_recipes(path=_write_recipes(tmp_path, SAMPLE))
    assert resolve_recipe(recipes, "does_not_exist", language="en") is None
    assert resolve_recipe(recipes, "", language="en") is None


def test_resolve_recipe_falls_back_when_language_field_missing(tmp_path):
    path = _write_recipes(
        tmp_path,
        """
        - id: en_only
          name_en: only english
          intents_en: [test]
          prompt_en: "english prompt"
        """,
    )
    recipes = load_camera_recipes(path=path)
    zh = resolve_recipe(recipes, "en_only", language="zh")
    # Falls back to the available language rather than returning blanks.
    assert zh["name"] == "only english"
    assert zh["prompt"] == "english prompt"
    assert zh["intents"] == ["test"]


def test_fill_recipe_slots_substitutes_subject():
    text = "slow eased push-in toward [the subject], focus locked on the subject"
    out = fill_recipe_slots(text, subject="the lighthouse keeper")
    assert "the lighthouse keeper" in out
    assert "[" not in out and "]" not in out


def test_fill_recipe_slots_resolves_possessive_and_direction():
    text = "subtle handheld drift following [the subject's] micro-movements to the [left/right]"
    out = fill_recipe_slots(text, subject="Mara")
    assert "Mara's micro-movements" in out
    # An either/or direction slot must collapse to one concrete direction, no bracket.
    assert "left" in out
    assert "[" not in out and "]" not in out


def test_fill_recipe_slots_uses_location_when_present():
    text = "one slow aerial drift gliding smoothly over [the location]"
    out = fill_recipe_slots(text, subject="the subject", location="the harbor town")
    assert "the harbor town" in out
    assert "[" not in out and "]" not in out


def test_fill_recipe_slots_chinese_slots():
    text = "缓慢横移镜头（向[左/右]），焦点在[主体]脸上"
    out = fill_recipe_slots(text, subject="船长", language="zh")
    assert "船长" in out
    assert "[" not in out and "]" not in out


def test_fill_recipe_slots_safety_net_strips_unknown_slots():
    text = "an unusual [mystery slot] and an [either/or] choice"
    out = fill_recipe_slots(text, subject="the subject")
    # Unknown single slot: brackets stripped, content preserved.
    assert "mystery slot" in out
    # Unknown either/or slot: first option kept.
    assert "either" in out
    assert "[" not in out and "]" not in out


def test_shipped_recipes_fully_resolve_with_no_brackets_left():
    """Every shipped recipe prompt, in both languages, must leave zero bracket tokens."""
    recipes = load_camera_recipes()
    for rid in recipes:
        for language in ("en", "zh"):
            resolved = resolve_recipe(recipes, rid, language=language)
            filled = fill_recipe_slots(
                resolved["prompt"], subject="the subject", location="the location", language=language
            )
            assert "[" not in filled and "]" not in filled, f"{rid}/{language} left a bracket: {filled!r}"


# Chaining connectors that would indicate a SECOND stacked move (authoring lint).
_STACK_CONNECTORS = (" then ", "; then", "随即", "再以", "然后", "接着", "紧接")


def test_library_recipes_are_single_move():
    recipes = load_camera_recipes()  # the real shipped library
    assert recipes, "expected the shipped recipe library to load"
    for rid, recipe in recipes.items():
        for field in ("prompt_en", "prompt_zh"):
            text = str(recipe.get(field, "")).lower()
            for connector in _STACK_CONNECTORS:
                assert connector.lower() not in text, f"{rid}.{field} stacks moves via {connector!r}"


def test_library_has_a_static_and_reserved_tier():
    recipes = load_camera_recipes()
    strengths = {str(r.get("motion_strength", "")) for r in recipes.values()}
    selections = {str(r.get("selection", "common")) for r in recipes.values()}
    assert "static" in strengths, "expected at least one static locked-off recipe"
    assert "reserved" in selections, "expected at least one reserved dramatic recipe"
    assert "common" in selections, "expected common calm-tier recipes"


def test_full_360_orbit_still_available_as_reserved_single_move():
    recipes = load_camera_recipes()
    orbits = [r for r in recipes.values() if "360" in str(r.get("prompt_zh", "")) + str(r.get("prompt_en", ""))]
    assert orbits, "360° orbit must remain available in the library"
    assert all(str(r.get("selection")) == "reserved" for r in orbits)


def test_resolve_recipe_exposes_selection(tmp_path):
    body = """
    - id: hold
      family: static
      name_en: locked-off hold
      name_zh: 锁定机位
      motion_strength: static
      selection: common
      intents_en: [dialogue]
      intents_zh: [对话]
      prompt_en: "locked-off camera, the subject carries the motion, slow performance"
      prompt_zh: "锁定机位，由主体承担运动，表演节奏缓慢"
    - id: sweep
      family: orbit
      name_en: slow 360 environment orbit
      name_zh: 缓慢360环境环绕
      motion_strength: high
      selection: reserved
      intents_en: [epic reveal]
      intents_zh: [史诗揭示]
      prompt_en: "one slow 360 orbit around the subject revealing the environment, eased"
      prompt_zh: "围绕主体缓慢环绕一周展示环境，缓入缓出"
    """
    path = tmp_path / "r.yaml"
    import textwrap
    path.write_text(textwrap.dedent(body))
    recipes = load_camera_recipes(path=path)
    assert resolve_recipe(recipes, "hold", language="en")["selection"] == "common"
    menu = recipe_menu(recipes, language="en")
    assert "sweep" in menu and "reserved" in menu.lower()
    assert "hold" in menu
