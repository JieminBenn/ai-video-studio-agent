"""Compact story context for generation prompt compilers.

Prompts get better when they know the story beat they are serving, but they should
not receive the whole project dump. This module extracts a small, deterministic
context packet from the project files: idea, plot summary, the matching script
scene, and neighboring shots.
"""

from __future__ import annotations

import json
from typing import Any

from .creative_brief import compact_creative_brief, load_creative_brief


def project_prompt_context(project, shot: dict[str, Any], *, shots: list[dict] | None = None) -> dict:
    """Build compact prompt context for ``shot`` from project-local files."""
    all_shots = list(shots) if shots is not None else _load_shots(project)
    return build_prompt_context(
        idea=_text(project.path("story", "idea.md")).strip(),
        plot=_json(project.path("story", "plot.json")),
        script=_json(project.path("story", "script.json")),
        shots=all_shots,
        shot=shot,
        language=project.model_config.get("language"),
        product_format=project.model_config.get("product_format"),
        creative_brief=compact_creative_brief(load_creative_brief(project)),
        genre=project.model_config.get("genre"),
    )


def build_prompt_context(
    *,
    idea: str = "",
    plot: dict[str, Any] | None = None,
    script: dict[str, Any] | None = None,
    shots: list[dict] | None = None,
    shot: dict[str, Any] | None = None,
    language: str | None = None,
    product_format: dict[str, Any] | None = None,
    creative_brief: dict[str, Any] | None = None,
    genre: str | None = None,
) -> dict[str, Any]:
    """Return a small dict safe to pass into pure prompt compilers."""
    plot = plot or {}
    script = script or {}
    shots = list(shots or [])
    shot = shot or {}
    prev_shot, next_shot = _neighbors(shots, shot)

    return _strip_empty({
        "idea": idea,
        "genre": (genre or "").strip(),
        "language": language,
        "format": _compact_format(product_format or {}),
        "creative_brief": creative_brief or {},
        "story": _compact_story(plot),
        "scene": _compact_scene(_script_scene(script, shot.get("scene"))),
        "previous_shot": _compact_shot(prev_shot),
        "next_shot": _compact_shot(next_shot),
    })


def _load_shots(project) -> list[dict]:
    path = project.path("storyboard", "shots.json")
    if not path.is_file():
        return []
    return json.loads(path.read_text()).get("shots", [])


def _text(path) -> str:
    return path.read_text() if path.is_file() else ""


def _json(path) -> Any:
    return json.loads(path.read_text()) if path.is_file() else {}


def _compact_story(plot: dict[str, Any]) -> dict[str, Any]:
    return _strip_empty({
        "logline": plot.get("logline"),
        "synopsis": plot.get("synopsis"),
        "themes": plot.get("themes", []),
    })


def _compact_scene(scene: dict[str, Any]) -> dict[str, Any]:
    return _strip_empty({
        "scene": scene.get("scene"),
        "heading": scene.get("heading"),
        "beats": list(scene.get("beats") or [])[:5],
        "dialogue": list(scene.get("dialogue") or [])[:5],
    })


def _compact_format(fmt: dict[str, Any]) -> dict[str, Any]:
    return _strip_empty({
        "name": fmt.get("name"),
        "label": fmt.get("label"),
        "structure": fmt.get("structure"),
        "pacing": fmt.get("pacing"),
    })


def _compact_shot(shot: dict[str, Any] | None) -> dict[str, Any]:
    if not shot:
        return {}
    keep = (
        "id", "scene", "description", "camera", "action", "dialogue",
        "camera_movement", "composition", "emotion", "continuity_notes",
        "duration_s", "characters",
    )
    return _strip_empty({key: shot.get(key) for key in keep})


def _neighbors(shots: list[dict], shot: dict[str, Any]) -> tuple[dict | None, dict | None]:
    shot_id = shot.get("id")
    for idx, candidate in enumerate(shots):
        same_id = shot_id and candidate.get("id") == shot_id
        same_object = not shot_id and candidate is shot
        if same_id or same_object:
            prev_shot = shots[idx - 1] if idx > 0 else None
            next_shot = shots[idx + 1] if idx + 1 < len(shots) else None
            return prev_shot, next_shot
    return None, None


def _script_scene(script: dict[str, Any], scene_no) -> dict[str, Any]:
    scenes = []
    for episode in script.get("episodes", []):
        scenes.extend(episode.get("scenes", []))
    for scene in scenes:
        if scene.get("scene") == scene_no:
            return scene
    return scenes[0] if len(scenes) == 1 else {}


def _strip_empty(data: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in data.items()
        if value not in (None, "", [], {})
    }
