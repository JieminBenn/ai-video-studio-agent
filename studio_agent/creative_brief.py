"""Versioned, file-backed creative direction for one Studio Agent project."""

from __future__ import annotations

import json
from typing import Any


VISUAL_FIELDS = (
    "tone",
    "visual_world",
    "camera_language",
    "light_texture",
    "character_treatment",
)

_SOURCES = {"user", "reference", "idea", "director_default"}
_CONFIDENCE = {"high", "medium", "low"}


def _choice(
    value: Any,
    *,
    source: str,
    confidence: str,
    locked: bool,
    reason: str,
) -> dict[str, Any]:
    return {
        "value": str(value or "").strip(),
        "source": source,
        "confidence": confidence,
        "locked": bool(locked),
        "reason": reason,
    }


def normalize_creative_brief(
    raw: Any,
    *,
    idea: str,
    model_config: dict[str, Any] | None,
    user_inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize legacy/model output and lock explicit kickoff choices."""
    data = dict(raw) if isinstance(raw, dict) else {}
    visual_raw = data.get("visual_direction")
    visual = dict(visual_raw) if isinstance(visual_raw, dict) else {}
    chosen = dict(user_inputs or {})
    normalized: dict[str, dict[str, Any]] = {}

    for key in VISUAL_FIELDS:
        if str(chosen.get(key) or "").strip():
            normalized[key] = _choice(
                chosen[key],
                source="user",
                confidence="high",
                locked=True,
                reason="Selected in the kickoff brief",
            )
            continue

        value = visual.get(key, "")
        if isinstance(value, dict):
            normalized[key] = _choice(
                value.get("value"),
                source=value.get("source") if value.get("source") in _SOURCES else "director_default",
                confidence=(
                    value.get("confidence")
                    if value.get("confidence") in _CONFIDENCE
                    else "medium"
                ),
                locked=bool(value.get("locked")),
                reason=str(value.get("reason") or "Normalized creative direction"),
            )
        else:
            normalized[key] = _choice(
                value,
                source="director_default",
                confidence="medium",
                locked=False,
                reason="Inferred from the idea and project configuration",
            )

    avoidances = data.get("hard_avoidances")
    if not isinstance(avoidances, list):
        avoidances = []
    user_avoidances = chosen.get("hard_avoidances")
    if isinstance(user_avoidances, str):
        user_avoidances = [user_avoidances]
    if not isinstance(user_avoidances, list):
        user_avoidances = []
    avoidances = list(dict.fromkeys(
        str(item).strip()
        for item in [*user_avoidances, *avoidances]
        if str(item).strip()
    ))
    config = dict(model_config or {})
    return {
        "version": 2,
        "source_idea": idea,
        "language": str(config.get("language") or "en"),
        "intent_summary": str(data.get("intent_summary") or idea),
        "visual_direction": normalized,
        "hard_avoidances": avoidances,
    }


def resolved_value(brief: dict[str, Any], key: str) -> str:
    value = (brief.get("visual_direction") or {}).get(key, "")
    if isinstance(value, dict):
        value = value.get("value")
    return str(value or "").strip()


def compact_creative_brief(brief: dict[str, Any]) -> dict[str, Any]:
    compact = {
        key: value
        for key in VISUAL_FIELDS
        if (value := resolved_value(brief, key))
    }
    compact["hard_avoidances"] = list(brief.get("hard_avoidances") or [])
    return compact


def load_creative_brief(project) -> dict[str, Any]:
    path = project.path("story", "creative_brief.json")
    return json.loads(path.read_text()) if path.is_file() else {}
