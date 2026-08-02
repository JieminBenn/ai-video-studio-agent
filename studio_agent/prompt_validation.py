"""Deterministic structural contract for motion-grid prompts.

Keyframe prompts are intentionally *not* pattern-validated. Scanning free prose for
camera/temporal/audio "motion" words guessed meaning from wording and mis-fired on ordinary
language ("pushes a contract", "tear tracks", "head bowed then lifting"), violating the
language-agnostic invariant (never regex prose to infer intent). The director already writes
STATIC keyframe prompts and a human reviews every prompt at the gate, so that guard cost more
(a false positive halts the whole batch) than it was worth (a stray motion word in a *still*
prompt is harmless — a still image cannot move).

The one check kept here is structural, not semantic: a motion-grid prompt must actually
enumerate its panels (``Panel 1..N`` / ``画格 N`` / ``分镜 N``), which downstream grid slicing
relies on. That matches a token the compiler itself emits — it is not reading meaning from prose.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptValidation:
    valid: bool
    errors: list[str]


def validate_grid_prompt(
    text: str,
    shot: dict,
    *,
    panel_count: int,
) -> PromptValidation:
    """A motion-grid prompt must name every panel it will be sliced into."""
    errors: list[str] = []
    for index in range(1, panel_count + 1):
        if not re.search(
            rf"(?:Panel|画格|分镜)\s*{index}\b",
            text,
            re.IGNORECASE,
        ):
            errors.append(f"missing panel {index}")
    return PromptValidation(not errors, list(dict.fromkeys(errors)))
