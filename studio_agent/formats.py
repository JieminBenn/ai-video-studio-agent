"""Product format presets.

Formats describe the intended shape of the project (short drama, short film,
series episode) without changing the pipeline itself. Stages read the selected
format from ``project.model_config`` and use it as creative guidance; files remain
the source of truth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ResolvedFormat:
    name: str
    spec: dict[str, Any]

    def to_project_config(self) -> dict[str, Any]:
        return {"name": self.name, **self.spec}


class UnknownFormatError(ValueError):
    def __init__(self, name: str, available: list[str]):
        self.name = name
        self.available = available
        super().__init__(f"unknown format '{name}'")


def resolve_format(config: dict, requested: str | None = None) -> ResolvedFormat:
    formats = dict(config.get("product_formats") or {})
    if not formats:
        formats = {
            "short_film": {
                "label": "Short film",
                "target_duration_s": 180,
                "min_duration_s": 120,
                "max_duration_s": 300,
            }
        }
    name = requested or config.get("default_format") or "short_film"
    if name not in formats:
        raise UnknownFormatError(name, sorted(formats))
    return ResolvedFormat(name=name, spec=dict(formats[name]))


def project_format(project) -> dict[str, Any]:
    fmt = dict(project.model_config.get("product_format") or {})
    if "name" not in fmt:
        fmt["name"] = project.model_config.get("format_name", "short_film")
    return fmt


def format_prompt_context(fmt: dict[str, Any]) -> str:
    """One-line JSON for prompts and fake-provider parsing."""
    return json.dumps(fmt or {}, ensure_ascii=False, sort_keys=True)


def genre_prompt(model_config: dict[str, Any] | None) -> str:
    """One locked-genre line for prompts (题材/类型先定), or "" when no genre is set.

    The tutorial's "风格先行,类型先定" rule: a locked genre must cascade into every
    downstream story and picture prompt. Empty genre preserves today's idea-only behavior.
    """
    genre = str((model_config or {}).get("genre") or "").strip()
    if not genre:
        return ""
    return (
        "GENRE (题材/类型 — lock all story, script, and visual choices to this genre): "
        f"{genre}\n"
    )


def genre_persona(model_config: dict[str, Any] | None) -> str:
    """A leading expert-identity qualifier for a stage persona, or "" when no genre.

    The tutorial's "是什么" step: don't ask a generic assistant — give the model a
    genre-specific expert identity so its answers come from the right discipline. This
    slots before the stage's role noun, e.g. ``"You are {genre_persona}a showrunner"``
    → ``"You are an award-winning 中国古代神话 specialist and a showrunner"``. Empty genre
    leaves the persona untouched.
    """
    genre = str((model_config or {}).get("genre") or "").strip()
    if not genre:
        return ""
    return f"an award-winning {genre} (题材) specialist and "


def target_duration_bounds(fmt: dict[str, Any]) -> tuple[float | None, float | None]:
    min_s = fmt.get("min_duration_s")
    max_s = fmt.get("max_duration_s")
    return (
        float(min_s) if min_s is not None else None,
        float(max_s) if max_s is not None else None,
    )


def format_mode(fmt: dict[str, Any] | None) -> str:
    """Pipeline mode for a resolved format: 'clip' for non-narrative short videos,
    'story' for the full narrative pipeline (the default when unset)."""
    return ((fmt or {}).get("mode") or "story")


def parse_length(value: Any) -> str | int:
    """Normalize a user length entry to 'auto' or a positive integer.

    Blank, "auto", or non-positive input means "use the preset default" → 'auto'.
    """
    text = str(value if value is not None else "auto").strip().lower()
    if text in ("", "auto"):
        return "auto"
    try:
        number = int(round(float(text)))
    except ValueError:
        return "auto"
    return number if number > 0 else "auto"


def apply_length(spec: dict[str, Any], mode: str, length: Any) -> dict[str, Any]:
    """Return a copy of ``spec`` with the user-chosen length applied for ``mode``.

    Clip mode sets ``clip_target_duration_s`` (seconds). Story mode interprets the
    value as minutes and overrides ``target/min/max_duration_s`` with proportional
    bounds. ``'auto'``/blank leaves the preset's own durations untouched.
    """
    out = dict(spec or {})
    value = parse_length(length)
    if mode == "clip":
        out["clip_target_duration_s"] = value
        return out
    if value == "auto":
        return out
    target = int(value) * 60
    out["target_duration_s"] = target
    out["min_duration_s"] = round(target * 0.7)
    out["max_duration_s"] = round(target * 1.4)
    return out
