"""Runtime filmmaking prompt skills.

These are product prompt/knowledge modules consumed by Studio Agent itself. They are
different from Codex workflow skills: this module loads small markdown files under the
repo-local ``skills/`` directory so stages can inject stable filmmaking structure into
LLM and generation prompts without hardcoding that knowledge into stage logic.
"""

from __future__ import annotations

import re
from pathlib import Path

SKILLS_ROOT = Path(__file__).resolve().parents[1] / "skills"
_SAFE_NAME = re.compile(r"^[a-z0-9_][a-z0-9_-]*$")


def load_prompt_skill(name: str, *, root: Path | None = None) -> str:
    """Return a packaged prompt skill's markdown, or ``""`` when it is absent."""
    if not _SAFE_NAME.match(name):
        return ""
    path = (root or SKILLS_ROOT) / f"{name}.md"
    if not path.is_file():
        return ""
    return path.read_text().strip()


def load_prompt_skills(names: list[str] | tuple[str, ...], *, root: Path | None = None) -> dict[str, str]:
    """Load multiple prompt skills, omitting missing names."""
    loaded: dict[str, str] = {}
    for name in names:
        text = load_prompt_skill(name, root=root)
        if text:
            loaded[name] = text
    return loaded
