"""Cinematography recipe library — vetted camera-movement templates.

These are a small, curated, bilingual set of expert camera-movement "recipes" keyed by
scene intent (e.g. a villain reveal wants a 360° orbit into a rapid push-in). They are a
*library*, not RAG: the set is finite, structured, and hand-editable, which fits the
"files are the source of truth" invariant far better than embeddings + vector search.

The storyboard planner picks a recipe ``id`` per shot, and the video prompt compiler
weaves the chosen recipe's detailed motion language into the prompt sent to the video
model. Chinese and English are first-class (invariant #10): every recipe carries ``*_zh``
and ``*_en`` fields and the resolver selects by project language with fallback.

This module performs only local file I/O and is otherwise pure, so it is unit-testable
offline alongside the rest of the deterministic core.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

RECIPES_PATH = Path(__file__).resolve().parent / "cinematography_recipes.yaml"

_LANGS = ("en", "zh")


def load_camera_recipes(*, path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load the recipe library keyed by ``id``; return ``{}`` when absent/empty."""
    target = path or RECIPES_PATH
    if not target.is_file():
        return {}
    data = yaml.safe_load(target.read_text()) or []
    recipes: dict[str, dict[str, Any]] = {}
    for entry in data:
        if isinstance(entry, dict) and entry.get("id"):
            recipes[str(entry["id"])] = entry
    return recipes


def _localized(recipe: dict[str, Any], field: str, language: str) -> Any:
    """Return ``field`` for ``language``, falling back to the other language."""
    primary = recipe.get(f"{field}_{language}")
    if primary:
        return primary
    for lang in _LANGS:
        value = recipe.get(f"{field}_{lang}")
        if value:
            return value
    return recipe.get(field)


def resolve_recipe(
    recipes: dict[str, dict[str, Any]], recipe_id: str | None, *, language: str = "en"
) -> dict[str, Any] | None:
    """Return a language-resolved view of one recipe, or ``None`` if unknown."""
    if not recipe_id:
        return None
    recipe = recipes.get(recipe_id)
    if recipe is None:
        return None
    return {
        "id": recipe.get("id", recipe_id),
        "family": recipe.get("family", ""),
        "motion_strength": recipe.get("motion_strength", ""),
        "selection": recipe.get("selection", "common"),
        "name": _localized(recipe, "name", language) or recipe_id,
        "intents": list(_localized(recipe, "intents", language) or []),
        "prompt": _localized(recipe, "prompt", language) or "",
    }


_SLOT_RE = re.compile(r"\[([^\[\]]+)\]")


def fill_recipe_slots(
    text: str,
    *,
    subject: str,
    location: str = "",
    language: str = "en",
) -> str:
    """Resolve a recipe's ``[bracket]`` placeholders into concrete shot language.

    Recipe prompt text ships slots like ``[the subject]``, ``[the subject's]``,
    ``[left/right]``, and ``[the location]`` so one template serves every shot. Nothing
    filled them before, so raw ``[the subject]`` tokens leaked verbatim into the model
    prompt. This substitutes the named slots from shot context and guarantees **no
    bracket token ever survives**: any remaining ``[a/b]`` collapses to its first option
    and any other ``[x]`` is stripped to ``x``.
    """
    subject = (subject or ("主体" if language == "zh" else "the subject")).strip()
    location = (location or ("该地点" if language == "zh" else "the location")).strip()
    other = "另一位角色" if language == "zh" else "the other character"

    # Named slots first (exact ``[inner]`` match); order-independent, no overlap.
    named = {
        "the subject": subject,
        "the subject's": f"{subject}'s",
        "the subject/environment": f"{subject} and the surrounding environment",
        "the subject/location": f"{subject} and the surrounding environment",
        "other party's": f"{other}'s",
        "other party": other,
        "the location": location,
        "主体": subject,
        "主体/环境": f"{subject}与周围环境",
        "对话对象": other,
        "地点": location,
    }
    for inner, value in named.items():
        text = text.replace(f"[{inner}]", value)

    # Safety net: any slot still present resolves deterministically and loses its brackets.
    def _resolve(match: re.Match[str]) -> str:
        return match.group(1).split("/")[0].strip()

    return _SLOT_RE.sub(_resolve, text)


def recipe_menu(recipes: dict[str, dict[str, Any]], *, language: str = "en") -> str:
    """Compact ``id — name — intents`` lines for the storyboard planner prompt."""
    lines: list[str] = []
    for recipe_id in recipes:
        resolved = resolve_recipe(recipes, recipe_id, language=language)
        if not resolved:
            continue
        intents = ", ".join(resolved["intents"])
        intent_label = "适用场景" if language == "zh" else "intents"
        marker = ""
        if resolved["selection"] == "reserved":
            marker = " [保留：仅在节拍配得上时]" if language == "zh" else " [reserved: only when the beat earns it]"
        lines.append(f"- {resolved['id']} — {resolved['name']}{marker} — {intent_label}: {intents}")
    return "\n".join(lines)
