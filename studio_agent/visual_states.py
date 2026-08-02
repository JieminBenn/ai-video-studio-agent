"""Shared, editable character visual-state contracts.

Stable identity belongs in the show Bible. Transformations, damage, disguises, aging,
effects, and other temporal changes belong in ``story/visual_state_changes.json``.
This module is deliberately pure so plot/concept stages can normalize one LLM response
without a second paid call and every later stage can consume the same file contract.
"""

from __future__ import annotations

import re
from typing import Any


_STATE_KINDS = {"base", "transient", "endpoint"}


def _state_id(value: object, *, fallback: str = "") -> str:
    text = str(value or "").strip().casefold().replace("_", "-")
    text = re.sub(r"[^\w\-]+", "-", text, flags=re.UNICODE)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or fallback


def _strings(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    output: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in output:
            output.append(text)
    return output


def normalize_visual_state_changes(
    raw: object,
    characters: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Return a safe, versioned visual-state document for known characters only."""
    known = {
        str(item.get("name") or "").strip().casefold(): str(item.get("name") or "").strip()
        for item in (characters or [])
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    }
    if not isinstance(raw, dict):
        return {"version": 1, "characters": []}
    if isinstance(raw.get("visual_state_changes"), dict):
        raw = raw["visual_state_changes"]

    normalized_plans: list[dict[str, Any]] = []
    for raw_plan in raw.get("characters") or []:
        if not isinstance(raw_plan, dict):
            continue
        requested_name = str(raw_plan.get("character") or "").strip()
        canonical_name = known.get(requested_name.casefold())
        if not canonical_name:
            continue

        states: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for index, raw_state in enumerate(raw_plan.get("states") or [], 1):
            if not isinstance(raw_state, dict):
                continue
            sid = _state_id(
                raw_state.get("id") or raw_state.get("label"),
                fallback=f"state-{index}",
            )
            if sid in seen_ids:
                continue
            seen_ids.add(sid)
            kind = str(raw_state.get("kind") or "transient").strip().casefold()
            if kind not in _STATE_KINDS:
                kind = "transient"
            appearance_changes = _strings(raw_state.get("appearance_changes"))
            material_endpoint = kind == "endpoint" and bool(appearance_changes)
            states.append({
                "id": sid,
                "label": str(raw_state.get("label") or sid).strip(),
                "kind": kind,
                "description": str(raw_state.get("description") or "").strip(),
                "appearance_changes": appearance_changes,
                "reference_required": kind == "base" or (
                    material_endpoint
                    and bool(raw_state.get("reference_required", True))
                ),
            })
        if not states:
            continue

        initial = _state_id(raw_plan.get("initial_state"))
        if initial not in seen_ids:
            initial = next(
                (state["id"] for state in states if state["kind"] == "base"),
                states[0]["id"],
            )

        transitions: list[dict[str, Any]] = []
        for raw_transition in raw_plan.get("transitions") or []:
            if not isinstance(raw_transition, dict):
                continue
            start = _state_id(raw_transition.get("from"))
            end = _state_id(raw_transition.get("to"))
            if start not in seen_ids or end not in seen_ids:
                continue
            ordered = [
                sid
                for value in raw_transition.get("ordered_state_ids") or []
                if (sid := _state_id(value)) in seen_ids
            ]
            if not ordered:
                ordered = [start, end]
            transitions.append({
                "from": start,
                "to": end,
                "trigger": str(raw_transition.get("trigger") or "").strip(),
                "ordered_state_ids": list(dict.fromkeys(ordered)),
                "preserve": _strings(raw_transition.get("preserve")),
            })

        normalized_plans.append({
            "character": canonical_name,
            "initial_state": initial,
            "states": states,
            "transitions": transitions,
        })

    return {"version": 1, "characters": normalized_plans}


def state_plan_for_character(document: object, name: str) -> dict[str, Any] | None:
    """Look up one character's normalized plan without assuming the file exists."""
    if not isinstance(document, dict):
        return None
    target = str(name or "").strip().casefold()
    for plan in document.get("characters") or []:
        if (
            isinstance(plan, dict)
            and str(plan.get("character") or "").strip().casefold() == target
        ):
            return plan
    return None
