"""Project language helpers.

Chinese and English are first-class. The language value is deliberately small
(``en`` or ``zh`` for now) so stages can preserve user intent without guessing on
every prompt.
"""

from __future__ import annotations

import re

_CJK_RE = re.compile(r"[\u3400-\u9fff]")


class UnknownLanguageError(ValueError):
    def __init__(self, name: str):
        self.name = name
        super().__init__(f"unknown language '{name}'")


def resolve_language(idea: str, requested: str | None = None) -> str:
    if requested in (None, "", "auto"):
        return "zh" if _CJK_RE.search(idea) else "en"
    normalized = requested.lower()
    aliases = {
        "english": "en",
        "en": "en",
        "chinese": "zh",
        "mandarin": "zh",
        "zh": "zh",
        "cn": "zh",
    }
    if normalized not in aliases:
        raise UnknownLanguageError(requested)
    return aliases[normalized]


def language_label(code: str) -> str:
    return {"zh": "Chinese", "en": "English"}.get(code, code)
