"""Rich, backward-compatible storyboard shot normalization."""

from __future__ import annotations

from typing import Any


def _unique(values) -> list[str]:
    output: list[str] = []
    for value in values or []:
        text = str(value).strip()
        if text and text not in output:
            output.append(text)
    return output


def _camera_angle(camera: str) -> str:
    value = camera.lower()
    if "low" in value:
        return "low angle"
    if "high" in value or "overhead" in value or "top-down" in value:
        return "high angle"
    return "eye-level unless the dramatic beat requires otherwise"


def _lens_intent(camera: str) -> str:
    value = camera.lower()
    if "close" in value:
        return "natural portrait perspective with readable contextual depth"
    if "wide" in value or "establish" in value:
        return "wide environmental perspective with controlled edge distortion"
    return "natural perspective preserving subject and environment"


def _fill(shot: dict[str, Any], key: str, value: str) -> None:
    if not str(shot.get(key) or "").strip():
        shot[key] = value


def normalize_shot_design(
    raw: dict[str, Any],
    *,
    prior_shot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    shot = dict(raw or {})
    camera = str(shot.get("camera") or "").strip()
    action = str(shot.get("action") or "").strip()
    composition = str(shot.get("composition") or "").strip()
    movement = str(shot.get("camera_movement") or "").strip()
    recipe = str(shot.get("camera_recipe") or "").strip()
    craft_ids = _unique(shot.get("craft_recipe_ids") or [])
    if recipe and recipe not in craft_ids:
        craft_ids.insert(0, recipe)

    _fill(shot, "dramatic_purpose", str(shot.get("emotion") or action).strip())
    _fill(
        shot,
        "story_advance",
        action or "advance the scene's next beat",
    )
    _fill(shot, "start_frame", composition or camera or "establish subject and space")
    _fill(
        shot,
        "end_frame",
        f"resolve on the visible result of: {action}" if action else "resolve on a clean cut point",
    )
    _fill(shot, "subject_blocking", action or "subject holds readable screen position")
    _fill(shot, "shot_size", camera or "medium shot")
    _fill(shot, "camera_angle", _camera_angle(camera))
    _fill(shot, "lens_intent", _lens_intent(camera))
    if movement:
        motivation = (
            "Preserve stillness so the performance carries the beat."
            if "lock" in movement.lower() or "static" in movement.lower()
            else f"Move only with the visible subject beat: {action or 'the action change'}."
        )
    else:
        motivation = f"Camera behavior follows the visible beat: {action or 'the action change'}."
    _fill(shot, "movement_motivation", motivation)
    _fill(
        shot,
        "lighting_state",
        "preserve the scene's motivated light direction and exposure continuity",
    )
    prior_id = str((prior_shot or {}).get("id") or "").strip()
    _fill(
        shot,
        "edit_relationship",
        f"continue screen direction from {prior_id}" if prior_id else "establish a clear entry state",
    )
    shot["camera_recipe"] = recipe
    shot["craft_recipe_ids"] = craft_ids
    return shot
