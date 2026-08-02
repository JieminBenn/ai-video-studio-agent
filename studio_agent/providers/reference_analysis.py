"""Shared helpers for vision-model reference analysis + description.

Multiple vision providers (OpenAI-compatible, Gemini, Anthropic) classify and describe
uploaded reference images. The system prompt, the per-image instruction, and the JSON
normalizer live here so every provider behaves identically (invariant #6: provider-agnostic).
"""

from __future__ import annotations

from typing import Any

REFERENCE_ANALYSIS_SYSTEM = (
    "You classify uploaded visual references for an AI filmmaking pipeline. Inspect "
    "the image itself and decide whether it primarily represents a character, a "
    "location/background, or a global visual style. Return strict JSON only."
)


def reference_analysis_text(aliases: list[str], user_note: str) -> str:
    return (
        "Analyze this one uploaded reference independently. "
        f"Its aliases are: {', '.join(aliases) or '(none)'}. "
        f"The user's note is: {user_note or '(none)'}. "
        "Return an object with target_type (character, location, style, or unknown), "
        "target_id (a concise character/location identifier, or global for style), "
        "confidence (0 to 1), reason, and visual_summary. Do not infer from upload "
        "order alone; use the visible content and the user's note."
    )


def normalize_reference_analysis(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {
            "target_type": "",
            "target_id": "",
            "confidence": 0.0,
            "reason": "vision model returned no structured analysis",
            "visual_summary": "",
        }
    target_type = str(value.get("target_type") or "").strip().lower()
    if target_type == "background":
        target_type = "location"
    if target_type not in {"character", "location", "style"}:
        target_type = ""
    try:
        confidence = max(0.0, min(1.0, float(value.get("confidence") or 0.0)))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "target_type": target_type,
        "target_id": "global" if target_type == "style" else str(value.get("target_id") or "").strip(),
        "confidence": confidence,
        "reason": str(value.get("reason") or "").strip(),
        "visual_summary": str(value.get("visual_summary") or "").strip(),
    }
