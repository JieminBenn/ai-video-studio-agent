"""Compatibility-safe normalization for rich character identity boards."""

from __future__ import annotations

import ast
from typing import Any


def coerce_board_dict(value: Any) -> dict[str, Any] | None:
    """Return a dict when ``value`` is a dict or a Python-dict-repr string."""
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                parsed = ast.literal_eval(s)
            except (ValueError, SyntaxError):
                return None
            if isinstance(parsed, dict):
                return parsed
    return None


def _humanize_key(key: str) -> str:
    return str(key).replace("_", " ").strip().capitalize()


def humanize_board_value(value: Any) -> str:
    """Render a board field as readable text instead of a raw dict repr."""
    as_dict = coerce_board_dict(value)
    if as_dict is not None:
        parts = []
        for key, sub in as_dict.items():
            text = humanize_board_value(sub).strip()
            if text:
                parts.append(f"{_humanize_key(key)}: {text}")
        return " · ".join(parts)
    if isinstance(value, list):
        return " · ".join(str(item).strip() for item in value if str(item).strip())
    return str(value or "").strip()


def _text(value: Any) -> str:
    """Stringify a scalar field, mapping JSON null to '' (never the literal 'None').

    A VLM that returns ``"wardrobe": null`` must not poison the model-sheet prompt with the
    string "None"; an absent/null field is simply empty.
    """
    return "" if value is None else str(value).strip()


def _list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    if value in (None, ""):
        return []
    return [value]


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _unique_strings(values: list[Any]) -> list[str]:
    output: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in output:
            output.append(text)
    return output


def coerce_aliases(values: Any) -> list[str]:
    """Normalize a ``prompt_aliases`` value to a deduped list of non-empty strings.

    Tolerates an LLM that returns alias *objects* (``{"alias"/"name"/"id": "..."}``) instead of
    bare strings — extract the label — so downstream ``", ".join(aliases)`` never sees a dict.
    Non-string/non-labelled and empty entries are dropped."""
    output: list[str] = []
    for item in values if isinstance(values, list) else []:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            text = str(
                item.get("alias") or item.get("name") or item.get("id") or ""
            ).strip()
        else:
            text = ""
        if text and text not in output:
            output.append(text)
    return output


def normalize_identity_board(
    raw: Any,
    *,
    name: str,
    canonical: dict[str, Any],
    references: list[str],
) -> dict[str, Any]:
    """Add rich invariants without changing legacy editable scalar fields."""
    data = dict(raw) if isinstance(raw, dict) else {}
    canonical_face = humanize_board_value(
        data.get("canonical_face") or canonical.get("description") or ""
    )
    canonical_body = humanize_board_value(data.get("canonical_body") or "")

    face = _dict(data.get("face"))
    face.setdefault("geometry", canonical_face)
    face.setdefault("distinctive_marks", [])
    face.setdefault("age_markers", "")
    face.setdefault("skin_details", "")

    body = _dict(data.get("body"))
    body.setdefault("proportions", canonical_body)
    body.setdefault("silhouette", canonical_body)
    body.setdefault("posture", "")
    body.setdefault("movement_signature", "")

    hair_raw = data.get("hair")
    hair_details = _dict(hair_raw)
    if not hair_details:
        hair_details = {"structure": str(hair_raw or "").strip()}
    hair_details.setdefault("color", "")
    hair_details.setdefault("grooming", "")

    wardrobe_raw = data.get("wardrobe")
    wardrobe_text = (
        _text(wardrobe_raw)
        if not isinstance(wardrobe_raw, dict)
        else _text(wardrobe_raw.get("summary"))
    )
    wardrobe_text = wardrobe_text or _text(canonical.get("wardrobe"))
    wardrobe_details = _dict(data.get("wardrobe_details"))
    if isinstance(wardrobe_raw, dict):
        wardrobe_details = {**wardrobe_raw, **wardrobe_details}
    wardrobe_details.setdefault("summary", wardrobe_text)
    wardrobe_details["immutable"] = _unique_strings(
        _list(wardrobe_details.get("immutable")) or _list(wardrobe_text)
    )
    wardrobe_details["variable"] = _unique_strings(
        _list(wardrobe_details.get("variable"))
    )
    wardrobe_details.setdefault("layers", [])
    wardrobe_details.setdefault("materials", [])
    wardrobe_details.setdefault("closures", [])

    aliases = _unique_strings(_list(data.get("prompt_aliases")) + [name])
    signature = str(data.get("identity_signature") or "").strip()
    if not signature:
        signature = "; ".join(
            value
            for value in (canonical_face, canonical_body, wardrobe_text)
            if value
        ) or name

    palette = _text(data.get("palette")) or _text(canonical.get("palette"))

    appearance_lock = humanize_board_value(data.get("appearance_lock") or "")
    if not appearance_lock:
        marks = ", ".join(_unique_strings(_list(face.get("distinctive_marks"))))
        immutable = ", ".join(wardrobe_details.get("immutable") or [])
        hair_text = " ".join(
            part for part in (hair_details.get("structure", ""), _text(hair_details.get("color"))) if part
        ).strip()
        segments = [
            f"Face: {canonical_face}" if canonical_face else "",
            f"Distinctive marks: {marks}" if marks else "",
            f"Skin: {_text(face.get('skin_details'))}" if _text(face.get("skin_details")) else "",
            f"Hair: {hair_text}" if hair_text else "",
            f"Build: {canonical_body}" if canonical_body else "",
            f"Wardrobe (immutable): {immutable}" if immutable else "",
            f"Palette/rendering: {palette}" if palette else "",
        ]
        appearance_lock = ". ".join(seg for seg in segments if seg).strip()

    return {
        **data,
        "identity_signature": signature,
        "canonical_face": canonical_face,
        "canonical_body": canonical_body,
        "face": face,
        "body": body,
        "hair": hair_raw if isinstance(hair_raw, str) else hair_details.get("structure", ""),
        "hair_details": hair_details,
        "wardrobe": wardrobe_text,
        "wardrobe_details": wardrobe_details,
        "palette": palette,
        "appearance_lock": appearance_lock,
        "hero_props": _unique_strings(_list(data.get("hero_props"))),
        "continuity_priority": _unique_strings(
            _list(data.get("continuity_priority"))
            or ["face", "silhouette", "hair", "wardrobe"]
        ),
        "allowed_variation": _unique_strings(
            _list(data.get("allowed_variation"))
        ),
        "do": _unique_strings(_list(data.get("do"))),
        "dont": _unique_strings(_list(data.get("dont"))),
        "prompt_aliases": aliases,
        "reference_bindings": _unique_strings(
            _list(data.get("reference_bindings")) + references
        ),
    }
