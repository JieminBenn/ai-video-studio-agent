"""Native reference-conditioned image provider (Gemini ``generate_content``).

The default real image provider for M1 consistency. Unlike Imagen, it accepts input
reference images (a character's bible reference + turnaround) and conditions the output on
them, so identity carries across shots (invariant #4). Model choice lives in config.yaml
(invariant #6): default ``gemini-3.1-flash-image`` (mixes up to ~14 refs), fallback
``gemini-2.5-flash-image`` (the flash image model, fewer refs).

Lazy client: neither the ``google-genai`` SDK nor ``GEMINI_API_KEY`` is needed to import or
construct this provider — only an actual ``generate()`` with the live API. The reference
count is guarded *before* any SDK import so the guard is testable offline. Install the SDK
with ``pip install -e ".[real]"``. The exact current model id is doc-verified against
ai.google.dev at build time.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from .base import Generation, ImageGen

DEFAULT_MODEL = "gemini-3.1-flash-image"
FALLBACK_MODEL = "gemini-2.5-flash-image"
# Conservative per-call reference-image limits by model family.
# Source: https://ai.google.dev/gemini-api/docs/image-generation (verified 2026-06-15):
# Gemini 3.x image models mix up to 14 reference images; 2.5 flash image takes fewer.
_MAX_REFS = {"gemini-3": 14, "gemini-2.5": 3}


def _default_max_refs(model: str) -> int:
    for prefix, limit in _MAX_REFS.items():
        if model.startswith(prefix):
            return limit
    return 3


def _first_image_bytes(resp):
    """Return the first inline image's bytes from a generate_content response, or None."""
    for candidate in getattr(resp, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in (getattr(content, "parts", None) or []):
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                return inline.data
    return None


def _no_image_diagnostics(resp) -> str:
    """A short, log-safe summary of why a response carried no image (for the raised error)."""
    bits = []
    feedback = getattr(resp, "prompt_feedback", None)
    if feedback:
        bits.append(f"prompt_feedback={feedback}")
    for candidate in getattr(resp, "candidates", None) or []:
        reason = getattr(candidate, "finish_reason", None)
        if reason is not None:
            bits.append(f"finish_reason={reason}")
        content = getattr(candidate, "content", None)
        for part in (getattr(content, "parts", None) or []):
            text = getattr(part, "text", None)
            if text:
                bits.append(f"text={text[:200]!r}")
        break
    return "; ".join(bits)


class GeminiImageGen(ImageGen):
    name = "gemini"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        cost_per_image: float = 0.04,
        api_key_env: str = "GEMINI_API_KEY",
        max_reference_images: int | None = None,
        max_attempts: int = 3,
        responder=None,
    ):
        self.model = model
        self.cost_per_image = cost_per_image
        self.api_key_env = api_key_env
        self.max_reference_images = (
            max_reference_images if max_reference_images is not None
            else _default_max_refs(model)
        )
        self.max_attempts = max(1, int(max_attempts))
        # Injectable for tests; the real responder calls the live SDK lazily.
        self._responder = responder
        self._client = None

    def _client_lazy(self):
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:  # pragma: no cover - optional extra
                raise RuntimeError(
                    "google-genai not installed — run `pip install -e \".[real]\"`."
                ) from exc
            key = os.environ.get(self.api_key_env)
            if not key:
                raise RuntimeError(
                    f"{self.api_key_env} not set — add it to .env (see `cli doctor`)."
                )
            self._client = genai.Client(api_key=key)
        return self._client

    def generate(
        self,
        prompt: str,
        *,
        out_path: str,
        reference_images: list[str] | None = None,
        **kwargs,
    ) -> Generation:
        refs = list(reference_images or [])
        # Guard BEFORE importing the SDK so it works offline (invariant: clear failures).
        if len(refs) > self.max_reference_images:
            raise ValueError(
                f"{self.model} accepts at most {self.max_reference_images} reference "
                f"images, but {len(refs)} were passed."
            )

        # Gemini image models intermittently return a text-only (no image) response;
        # retry a few times so a single transient miss never aborts a paid run.
        start = time.time()
        last_diag = ""
        for _ in range(self.max_attempts):
            resp = self._respond(prompt, refs)
            image_bytes = _first_image_bytes(resp)
            if image_bytes:
                elapsed = round(time.time() - start, 2)
                path = Path(out_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(image_bytes)
                return Generation(
                    content=str(path),
                    provider=self.name,
                    model=self.model,
                    cost_usd=self.cost_per_image,
                    seconds=elapsed,
                    meta={"seed": kwargs.get("seed"), "reference_images": refs},
                )
            last_diag = _no_image_diagnostics(resp)

        raise RuntimeError(
            f"Gemini generate_content returned no image part after {self.max_attempts} "
            f"attempt(s). {last_diag}".strip()
        )

    def _respond(self, prompt: str, refs: list[str]):
        if self._responder is not None:
            return self._responder(prompt=prompt, reference_images=refs)

        from google.genai import types

        # The text prompt leads, followed by each reference image as an inline part. The
        # mime type is sniffed from the extension so PNG and JPEG references both work.
        contents = [prompt]
        for ref in refs:
            mime = "image/jpeg" if ref.lower().endswith((".jpg", ".jpeg")) else "image/png"
            contents.append(
                types.Part.from_bytes(data=Path(ref).read_bytes(), mime_type=mime)
            )
        return self._client_lazy().models.generate_content(
            model=self.model, contents=contents
        )
