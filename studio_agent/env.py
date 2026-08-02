"""Local environment loading for API keys.

Secrets live in the repo-local `.env` file, which is gitignored. Values are loaded into
``os.environ`` for provider implementations, but diagnostics only report whether a key
is present; they never print the secret value.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

DEFAULT_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

KNOWN_API_KEYS = [
    "OPENAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "GEMINI_API_KEY",
    "ANTHROPIC_API_KEY",
    "XAI_API_KEY",
    "FAL_KEY",
    "REPLICATE_API_TOKEN",
    "ELEVENLABS_API_KEY",
    "DOUBAO_API_KEY",
    "ARK_API_KEY",
    "BYTEPLUS_ARK_API_KEY",
    "VLLM_API_KEY",
]

_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_dotenv(text: str) -> dict[str, str]:
    """Parse simple KEY=VALUE lines from a dotenv file."""
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not _KEY_RE.match(key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        values[key] = value
    return values


def load_dotenv(path: str | Path = DEFAULT_ENV_PATH, *, override: bool = False) -> list[str]:
    """Load env values from ``path``. Existing shell env vars win by default."""
    env_path = Path(path)
    if not env_path.is_file():
        return []

    loaded = []
    for key, value in parse_dotenv(env_path.read_text()).items():
        if override or key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def api_key_status(keys: list[str] | None = None) -> dict[str, bool]:
    """Return whether each known key is set, without exposing values."""
    return {key: bool(os.environ.get(key)) for key in (keys or KNOWN_API_KEYS)}
