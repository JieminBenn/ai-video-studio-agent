"""Cached, file-backed creative decisions used by stage preflights."""

from __future__ import annotations

import json
from typing import Any

from .stages.base import StagePreflight


_ALTERNATIVES = {
    "camera_language": "energetic and immediate",
    "character_treatment": "iconic and carefully composed",
}


def build_decision_prompt(*, stage: str, gap: dict[str, Any], project) -> str:
    language = project.model_config.get("language", "en")
    return (
        "[task:creative_decision]\n"
        "You are a film director. Ask one concise question about intended feeling, not "
        "technical camera jargon. Return JSON with question, why_it_matters, evidence, "
        "choices (two or three objects with value, label, description), and default.\n"
        f"IDEA: {project.idea}\n"
        f"LANGUAGE: {language}\n"
        f"STAGE: {stage}\n"
        f"GAP: {json.dumps(gap, ensure_ascii=False, sort_keys=True)}\n"
    )


def _fallback_choices(gap: dict[str, Any]) -> list[dict[str, str]]:
    dimension = str(gap.get("dimension") or "creative_direction")
    default = str(gap.get("default") or "use the director's restrained choice")
    alternative = _ALTERNATIVES.get(dimension, "a bolder contrasting choice")
    if alternative == default:
        alternative = "a quieter contrasting choice"
    return [
        {
            "value": default,
            "label": default,
            "description": "The director's recommended interpretation.",
        },
        {
            "value": alternative,
            "label": alternative,
            "description": "A meaningful contrasting interpretation.",
        },
    ]


def normalize_decision(
    raw: Any,
    *,
    stage: str,
    gap: dict[str, Any],
) -> dict[str, Any]:
    data = dict(raw) if isinstance(raw, dict) else {}
    choices = []
    raw_choices = data.get("choices") if isinstance(data.get("choices"), list) else []
    for item in raw_choices:
        if not isinstance(item, dict) or not str(item.get("value") or "").strip():
            continue
        value = str(item["value"]).strip()
        choices.append({
            "value": value,
            "label": str(item.get("label") or value).strip(),
            "description": str(item.get("description") or "").strip(),
        })
    if len(choices) not in (2, 3):
        choices = _fallback_choices(gap)

    allowed = [item["value"] for item in choices]
    requested_default = str(data.get("default") or gap.get("default") or "").strip()
    default = requested_default if requested_default in allowed else allowed[0]
    evidence = data.get("evidence") if isinstance(data.get("evidence"), list) else []
    dimension = str(gap.get("dimension") or "creative_direction")
    return {
        "version": 1,
        "stage": stage,
        "dimension": dimension,
        "question": str(
            data.get("question")
            or f"Which feeling should guide the {dimension.replace('_', ' ')}?"
        ).strip(),
        "why_it_matters": str(data.get("why_it_matters") or gap.get("why") or "").strip(),
        "evidence": [str(item) for item in evidence if str(item).strip()],
        "choices": choices,
        "default": default,
        "resolution": None,
    }


def decision_preflight(
    project,
    providers,
    *,
    stage: str,
    gap: dict[str, Any],
    auto: bool,
) -> StagePreflight:
    path = project.path("story", "decisions", f"{stage}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        data = json.loads(path.read_text())
    else:
        project.assert_budget_available()
        gen = providers.llm.complete_json(
            build_decision_prompt(stage=stage, gap=gap, project=project)
        )
        project.add_generation_cost(stage=stage, generation=gen)
        data = normalize_decision(gen.content, stage=stage, gap=gap)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")

    if data.get("resolution"):
        return StagePreflight()
    if auto:
        data["resolution"] = {
            "value": data["default"],
            "source": "director_default",
            "locked": False,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        return StagePreflight()
    return StagePreflight(
        decision_required=True,
        request_path=str(path),
        message=str(data.get("question") or "Creative decision required"),
    )


def resolve_decision(project, *, stage: str, choice: str) -> dict[str, Any]:
    path = project.path("story", "decisions", f"{stage}.json")
    data = json.loads(path.read_text())
    allowed = [str(item["value"]) for item in data.get("choices", [])]
    if choice not in allowed and choice != data.get("default"):
        raise ValueError(f"unknown decision choice for {stage}: {choice}")
    data["resolution"] = {"value": choice, "source": "user", "locked": True}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    return data
