"""Normalize one global short-video idea into a few coherent provider-safe clips."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


_DURATION_CHOICES = (15, 30, 60)
_COMPLEXITY_TIERS = ("simple", "moderate", "complex")
_PLAN_MODES = ("default", "manual", "auto")


@dataclass(frozen=True)
class ClipPlan:
    """How clip mode decides the number of clips and each clip's length.

    ``default`` — one clip whose length the video model decides (``seconds=None``).
    ``manual``  — the user fixed ``count`` clips of ``seconds`` each (``None`` seconds
    still defers each clip's length to the model).
    ``auto``    — the LLM's complexity judgment sets a total that is split into the
    fewest provider-legal clips (today's legacy behavior); ``total_s`` pins that total
    when a legacy project requested a fixed runtime instead of ``auto``.
    """

    mode: str = "default"
    count: int = 1
    seconds: int | None = None
    total_s: int | None = None


def clamp_clip_duration(seconds: Any, *, min_s: int = 4, max_s: int = 15) -> int | None:
    """Clamp a per-clip duration to the provider's legal integer range.

    Blank/``None``/non-numeric input returns ``None``, meaning "let the video model's
    own default length govern" — the adapters translate that to their omit-duration
    form (Ark ``-1``, fal ``"auto"``, Veo drops the field).
    """
    if seconds is None:
        return None
    if isinstance(seconds, str) and not seconds.strip():
        return None
    try:
        value = int(round(float(seconds)))
    except (TypeError, ValueError):
        return None
    maximum = max(1, _positive_int(max_s, 15))
    minimum = min(maximum, max(1, _positive_int(min_s, 4)))
    return min(maximum, max(minimum, value))


def resolve_clip_plan(model_config: dict, fmt: dict | None = None) -> ClipPlan:
    """Resolve the project's clip plan from model_config with format-spec fallbacks.

    An explicit ``clip_plan_mode`` always wins. Projects created before the key existed
    keep their old behavior: ``clip_target_duration_s == "auto"`` stays LLM-planned, a
    numeric target stays a fixed-total split, and otherwise the legacy
    ``clip_count``/``clip_seconds`` pair applies.
    """
    fmt = fmt or {}

    def _setting(key: str) -> Any:
        value = model_config.get(key)
        return value if value is not None else fmt.get(key)

    count = _positive_int(_setting("clip_count"), 1)
    seconds = _int_or_none(_setting("clip_seconds"))
    # clip_plan_mode is model_config-only: the CLI/dashboard write it for every new
    # project. A format-spec fallback would leak the new default into legacy-shaped
    # projects that expressed intent through clip_count/clip_seconds alone.
    mode = str(model_config.get("clip_plan_mode") or "").strip().lower()
    if mode in _PLAN_MODES:
        if mode == "default":
            return ClipPlan(mode="default", count=1, seconds=None)
        if mode == "manual":
            return ClipPlan(mode="manual", count=count, seconds=seconds)
        return ClipPlan(mode="auto")

    # Legacy projects (no clip_plan_mode): match the old stage code exactly — the
    # requested duration was read from model_config only (the CLI/dashboard always
    # wrote it), never from the format spec, whose "auto" default must not override
    # an explicit legacy clip_count/clip_seconds pair.
    requested = model_config.get("clip_target_duration_s")
    if str(requested or "").strip().lower() == "auto":
        return ClipPlan(mode="auto")
    if requested not in (None, ""):
        return ClipPlan(mode="auto", total_s=_positive_int(requested, 15))
    return ClipPlan(mode="manual", count=count, seconds=seconds or 15)


def _int_or_none(value: Any) -> int | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        return max(1, int(round(float(value))))
    except (TypeError, ValueError):
        return None


def _positive_int(value: Any, default: int) -> int:
    try:
        return max(1, int(round(float(value))))
    except (TypeError, ValueError):
        return default


def _beat_duration_total(raw: dict) -> int:
    """Sum per-beat ``duration_s`` (0 when no beat declares one)."""
    beats = raw.get("beats")
    if not isinstance(beats, list):
        return 0
    total = 0
    found = False
    for beat in beats:
        if isinstance(beat, dict) and beat.get("duration_s") is not None:
            total += _positive_int(beat.get("duration_s"), 0)
            found = True
    return total if found else 0


def balanced_clip_durations(
    total_s: Any,
    *,
    max_clip_s: int = 15,
    min_clip_s: int = 4,
) -> list[int]:
    """Split a total into the fewest clips, keeping their lengths nearly equal."""
    maximum = max(1, _positive_int(max_clip_s, 15))
    minimum = min(maximum, max(1, _positive_int(min_clip_s, 4)))
    total = max(minimum, _positive_int(total_s, maximum))
    count = max(1, math.ceil(total / maximum))
    base, remainder = divmod(total, count)
    durations = [base + (1 if index < remainder else 0) for index in range(count)]
    return durations


def _complexity_cap(raw: dict, *, max_clip_s: int, ceiling_s: int) -> int | None:
    """Hard runtime cap implied by the LLM's complexity judgment, or None when absent.

    simple -> one provider-legal clip; moderate -> two; complex -> the ceiling. Tiers are
    derived from the provider's per-clip max, never hardcoded (invariant #6)."""
    tier = str(raw.get("complexity") or "").strip().lower()
    if tier not in _COMPLEXITY_TIERS:
        return None
    cap = max(1, _positive_int(max_clip_s, 15))
    if tier == "moderate":
        return min(ceiling_s, cap * 2)
    if tier == "complex":
        return ceiling_s
    return cap


def _target_duration(
    raw: dict,
    requested_duration: Any,
    *,
    legacy_count: Any,
    legacy_seconds: Any,
    max_clip_s: int = 15,
    ceiling_s: int = 60,
    floor_s: int = 4,
) -> int:
    if str(requested_duration or "").lower() == "auto":
        cap = _complexity_cap(raw, max_clip_s=max_clip_s, ceiling_s=ceiling_s)
        ceiling = min(ceiling_s, cap) if cap is not None else ceiling_s
        beat_total = _beat_duration_total(raw)
        if beat_total:
            return max(floor_s, min(ceiling, beat_total))
        recommended = _positive_int(
            raw.get("recommended_duration_s"),
            30 if raw.get("intent_class") in {"transformation", "reveal", "micro-arc"} else 15,
        )
        bucketed = next((choice for choice in _DURATION_CHOICES if recommended <= choice), 60)
        return min(bucketed, cap) if cap is not None else bucketed
    if requested_duration not in (None, ""):
        value = _positive_int(requested_duration, 15)
        return value if value in _DURATION_CHOICES else min(
            _DURATION_CHOICES, key=lambda choice: abs(choice - value)
        )
    count = _positive_int(legacy_count, 1)
    seconds = _positive_int(legacy_seconds, 15)
    return count * seconds


def _normalize_beats(raw_beats: Any) -> list[dict]:
    beats: list[dict] = []
    for index, item in enumerate(raw_beats or [], start=1):
        if not isinstance(item, dict):
            item = {"action": str(item)}
        beat = dict(item)
        beat["id"] = str(beat.get("id") or f"beat-{index:02d}").strip()
        beat["action"] = str(
            beat.get("action") or beat.get("description") or beat["id"]
        ).strip()
        for field in ("start_states", "end_states"):
            values = beat.get(field)
            beat[field] = {
                str(name): str(state)
                for name, state in (values.items() if isinstance(values, dict) else [])
                if str(name).strip() and str(state).strip()
            }
        beats.append(beat)
    return beats or [{
        "id": "motion",
        "action": "Perform one clear, intentional visual action",
        "start_states": {},
        "end_states": {},
    }]


def _beat_groups(beats: list[dict], count: int) -> list[list[dict]]:
    if len(beats) >= count:
        base, remainder = divmod(len(beats), count)
        groups: list[list[dict]] = []
        cursor = 0
        for index in range(count):
            size = base + (1 if index < remainder else 0)
            groups.append(beats[cursor:cursor + size])
            cursor += size
        return groups
    return [[beats[min((index * len(beats)) // count, len(beats) - 1)]] for index in range(count)]


def _first_states(beats: list[dict], fallback: dict[str, str]) -> dict[str, str]:
    for beat in beats:
        if beat["start_states"]:
            return dict(beat["start_states"])
        if beat["end_states"]:
            return dict(beat["end_states"])
    return dict(fallback)


def _last_states(beats: list[dict], fallback: dict[str, str]) -> dict[str, str]:
    for beat in reversed(beats):
        if beat["end_states"]:
            return dict(beat["end_states"])
        if beat["start_states"]:
            return dict(beat["start_states"])
    return dict(fallback)


def normalize_visual_sequence(
    raw: Any,
    *,
    requested_duration: Any = "auto",
    legacy_count: Any = 1,
    legacy_seconds: Any = 15,
    max_clip_s: int = 15,
    min_clip_s: int = 4,
    ceiling_s: int = 60,
    plan: ClipPlan | None = None,
) -> dict:
    """Return an editable global beat plan plus the fewest coherent clip segments.

    With a ``plan``, ``default``/``manual`` modes pin the clip count (and optional
    per-clip seconds; ``None`` = model-default length) while ``auto`` keeps the
    LLM-complexity split. Without one, the legacy keyword arguments govern.
    """
    source = dict(raw) if isinstance(raw, dict) else {}
    durations: list[int | None]
    if plan is not None and plan.mode in ("default", "manual"):
        seconds = clamp_clip_duration(plan.seconds, min_s=min_clip_s, max_s=max_clip_s)
        durations = [seconds] * max(1, plan.count)
    else:
        if plan is not None:
            requested_duration = plan.total_s if plan.total_s is not None else "auto"
        target = _target_duration(
            source,
            requested_duration,
            legacy_count=legacy_count,
            legacy_seconds=legacy_seconds,
            max_clip_s=max_clip_s,
            ceiling_s=ceiling_s,
            floor_s=min_clip_s,
        )
        durations = list(balanced_clip_durations(
            target, max_clip_s=max_clip_s, min_clip_s=min_clip_s
        ))
    beats = _normalize_beats(source.get("beats"))
    groups = _beat_groups(beats, len(durations))
    segments = []
    carried_states: dict[str, str] = {}
    for index, (duration, group) in enumerate(zip(durations, groups), start=1):
        start_states = _first_states(group, carried_states)
        end_states = _last_states(group, start_states)
        segments.append({
            "id": f"segment-{index:02d}",
            "duration_s": duration,
            "visual_beats": group,
            "start_states": start_states,
            "end_states": end_states,
        })
        carried_states = end_states
    known_durations = [value for value in durations if value is not None]
    return {
        "version": 1,
        "intent_class": str(source.get("intent_class") or "showcase"),
        "target_duration_s": sum(known_durations) if known_durations else None,
        "clip_durations": durations,
        "beats": beats,
        "segments": segments,
    }


def _beat_seconds(beat: dict, default: int) -> int:
    return _positive_int(beat.get("duration_s"), default)


def _beat_has_dialogue(beat: dict) -> bool:
    dialogue = beat.get("dialogue")
    return isinstance(dialogue, list) and any(
        isinstance(line, dict) and str(line.get("line") or "").strip()
        for line in dialogue
    )


def _group_narration(beats: list[dict]) -> str:
    for beat in beats:
        text = str(beat.get("narration") or "").strip()
        if text:
            return text
    return ""


def plan_scene_segments(
    beats: Any,
    *,
    max_clip_s: int = 15,
    min_clip_s: int = 4,
) -> list[dict]:
    """Group a scene's ordered beats into clip segments.

    Dialogue beats stay atomic (one segment each) so lip-sync stays reliable; contiguous
    narration-only / action-only beats merge into the fewest clips no longer than
    ``max_clip_s``. A small scene yields a single short clip; a rich scene yields a few.
    """
    maximum = max(1, _positive_int(max_clip_s, 15))
    minimum = min(maximum, max(1, _positive_int(min_clip_s, 4)))
    normalized = _normalize_beats(beats)
    segments: list[dict] = []
    buffer: list[dict] = []
    carried: dict[str, str] = {}

    def emit(group: list[dict], duration: int) -> None:
        nonlocal carried
        start_states = _first_states(group, carried)
        end_states = _last_states(group, start_states)
        dialogue = [
            line
            for beat in group
            for line in (beat.get("dialogue") or [])
            if isinstance(line, dict) and str(line.get("line") or "").strip()
        ]
        segments.append({
            "id": f"segment-{len(segments) + 1:02d}",
            "duration_s": duration,
            "visual_beats": group,
            "start_states": start_states,
            "end_states": end_states,
            "dialogue": dialogue,
            "narration": _group_narration(group),
        })
        carried = end_states

    def flush() -> None:
        if not buffer:
            return
        total = sum(_beat_seconds(beat, minimum) for beat in buffer)
        durations = balanced_clip_durations(total, max_clip_s=maximum, min_clip_s=minimum)
        groups = _beat_groups(buffer, len(durations))
        for duration, group in zip(durations, groups):
            emit(group, duration)
        buffer.clear()

    for beat in normalized:
        if _beat_has_dialogue(beat):
            flush()
            emit([beat], min(maximum, max(minimum, _beat_seconds(beat, minimum))))
        else:
            buffer.append(beat)
    flush()
    return segments
