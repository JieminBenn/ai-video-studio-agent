"""Real video QC and style profiling via Claude vision models (invariants #6, #9).

``AnthropicVLMCheck`` is the M1 replacement for ``FakeVLMCheck``: it samples frames from a
generated clip, sends them to Claude with the shot's compiled prompt, and asks for a
structured per-dimension QC verdict (prompt adherence, identity drift, motion/anatomy,
artifacts, audio mismatch, safety). The verdict is normalized into the same report shape
every stage already consumes, so swapping fake -> real is a config change (invariant #6).

``AnthropicStyleProfiler`` turns a free-form style description and/or a single reference
image into the pipeline's canonical style dict via Claude. It mirrors the same lazy,
injectable-responder pattern as ``AnthropicVLMCheck`` so the deterministic core is
unit-tested with no SDK or network.

Both classes are lazy: neither the ``anthropic`` SDK nor ``ANTHROPIC_API_KEY`` is needed to
import or construct them — only an actual live call. Install the SDK with
``pip install -e ".[real]"``.

Model id and pricing are doc-verified for ``claude-opus-4-8`` (vision + structured
outputs; $5 / $25 per 1M input/output tokens).
"""

from __future__ import annotations

import base64
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import os

from .base import QC_DIMENSIONS, Generation, ReferenceAnalyzer, StyleProfiler, VLMCheck, qc_report_status
from .openai_compatible import parse_json_content
from .reference_analysis import (
    REFERENCE_ANALYSIS_SYSTEM,
    normalize_reference_analysis,
    reference_analysis_text,
)
from ..style import normalize_style_dict
from .style_prompt import STYLE_SYSTEM, style_user_text

DEFAULT_MODEL = "claude-opus-4-8"
# claude-opus-4-8 list price: $5.00 / 1M input, $25.00 / 1M output → per-1k rates.
DEFAULT_COST_PER_1K_INPUT_USD = 0.005
DEFAULT_COST_PER_1K_OUTPUT_USD = 0.025

DEFAULT_STYLE_MODEL = DEFAULT_MODEL
DEFAULT_STYLE_COST_PER_1K_INPUT_USD = DEFAULT_COST_PER_1K_INPUT_USD
DEFAULT_STYLE_COST_PER_1K_OUTPUT_USD = DEFAULT_COST_PER_1K_OUTPUT_USD

_SYSTEM = (
    "You are a strict video QC reviewer for an AI film pipeline. You are shown ordered "
    "frames sampled from one generated shot plus the full story-aware QC context it must "
    "satisfy. Judge story and shot intent before surface polish. Mark a dimension as not "
    "passed only when there is a clear, defensible problem; cite a frame/timestamp when "
    "you can."
)

# Structured-output schema so the response is guaranteed parseable JSON.
_SCHEMA = {
    "type": "object",
    "properties": {
        "checks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "dimension": {"type": "string", "enum": list(QC_DIMENSIONS)},
                    "passed": {"type": "boolean"},
                    "severity": {"type": "string", "enum": ["none", "low", "medium", "high"]},
                    "detail": {"type": "string"},
                    "timestamp": {"type": "string"},
                },
                "required": ["dimension", "passed", "severity", "detail"],
                "additionalProperties": False,
            },
        },
        "summary": {"type": "string"},
    },
    "required": ["checks", "summary"],
    "additionalProperties": False,
}


class AnthropicVLMCheck(VLMCheck, ReferenceAnalyzer):
    name = "anthropic"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        max_frames: int = 4,
        max_tokens: int = 2048,
        cost_per_1k_input_usd: float = DEFAULT_COST_PER_1K_INPUT_USD,
        cost_per_1k_output_usd: float = DEFAULT_COST_PER_1K_OUTPUT_USD,
        api_key_env: str = "ANTHROPIC_API_KEY",
        responder: Callable[..., dict[str, Any]] | None = None,
        reference_responder: Callable[..., dict[str, Any]] | None = None,
    ):
        self.model = model
        self.max_frames = max_frames
        self.max_tokens = max_tokens
        self.cost_per_1k_input_usd = cost_per_1k_input_usd
        self.cost_per_1k_output_usd = cost_per_1k_output_usd
        self.api_key_env = api_key_env
        self._responder = responder or _default_responder
        self._reference_responder = reference_responder or _default_reference_responder

    def analyze(
        self,
        image_path: str,
        *,
        aliases: list[str] | None = None,
        user_note: str = "",
    ) -> Generation:
        image = Path(image_path)
        if not image.is_file():
            raise FileNotFoundError(f"reference image not found: {image_path}")
        start = time.time()
        result = self._reference_responder(
            image_paths=[str(image)],
            prompt=reference_analysis_text(aliases or [], user_note),
            system=REFERENCE_ANALYSIS_SYSTEM,
            model=self.model,
            max_tokens=min(self.max_tokens, 1024),
            api_key_env=self.api_key_env,
        )
        analysis = normalize_reference_analysis(parse_json_content(result.get("text", "")))
        usage = result.get("usage") or {}
        return Generation(
            content=analysis,
            provider=self.name,
            model=str(result.get("model") or self.model),
            cost_usd=self._cost(usage),
            seconds=round(time.time() - start, 3),
            meta={"usage": usage, "input": "reference_image"},
        )

    def describe(
        self,
        image_paths: list[str],
        *,
        prompt: str,
        language: str = "en",
    ) -> Generation:
        images = [str(path) for path in image_paths if Path(path).is_file()]
        if not images:
            raise FileNotFoundError("no reference image found to describe")
        start = time.time()
        result = self._reference_responder(
            image_paths=images,
            prompt=prompt,
            system=None,
            model=self.model,
            max_tokens=self.max_tokens,
            api_key_env=self.api_key_env,
        )
        content = parse_json_content(result.get("text", ""))
        usage = result.get("usage") or {}
        return Generation(
            content=content if isinstance(content, dict) else {},
            provider=self.name,
            model=str(result.get("model") or self.model),
            cost_usd=self._cost(usage),
            seconds=round(time.time() - start, 3),
            meta={"usage": usage, "input": "reference_describe"},
        )

    def revise(
        self,
        image_paths: list[str],
        *,
        prompt: str,
        language: str = "en",
    ) -> Generation:
        images = [str(path) for path in image_paths if Path(path).is_file()]
        if not images:
            raise FileNotFoundError("no reference image found to revise")
        start = time.time()
        result = self._reference_responder(
            image_paths=images,
            prompt=prompt,
            system=None,
            model=self.model,
            max_tokens=self.max_tokens,
            api_key_env=self.api_key_env,
        )
        usage = result.get("usage") or {}
        return Generation(
            content=result.get("text", ""),
            provider=self.name,
            model=str(result.get("model") or self.model),
            cost_usd=self._cost(usage),
            seconds=round(time.time() - start, 3),
            meta={"usage": usage, "input": "reference_revise"},
        )

    def review(self, clip_path: str, *, prompt: str, **kwargs: Any) -> Generation:
        clip = Path(clip_path)
        if not clip.is_file():
            raise FileNotFoundError(f"VLM QC clip not found: {clip_path}")

        start = time.time()
        result = self._responder(
            clip_path=str(clip),
            prompt=prompt,
            model=self.model,
            max_frames=self.max_frames,
            max_tokens=self.max_tokens,
            api_key_env=self.api_key_env,
        )
        report = _normalize(result.get("verdict") or {}, clip, prompt)
        usage = result.get("usage") or {}
        cost = self._cost(usage)
        return Generation(
            content=report,
            provider=self.name,
            model=str(result.get("model") or self.model),
            cost_usd=cost,
            seconds=round(time.time() - start, 2),
            meta={"usage": usage, "frames_sampled": self.max_frames},
        )

    def _cost(self, usage: dict[str, Any]) -> float:
        in_tok = float(usage.get("input_tokens") or 0)
        out_tok = float(usage.get("output_tokens") or 0)
        return round(
            in_tok / 1000 * self.cost_per_1k_input_usd
            + out_tok / 1000 * self.cost_per_1k_output_usd,
            6,
        )


class AnthropicStyleProfiler(StyleProfiler):
    """Profile a free-form style (text and/or one image) into a style dict via Claude.

    Mirrors ``AnthropicVLMCheck``: the live ``messages.create`` call is lazy and lives in an
    injectable responder, so the deterministic core is unit-tested with no SDK or network.
    """

    name = "anthropic-style"

    def __init__(
        self,
        model: str = DEFAULT_STYLE_MODEL,
        *,
        api_key_env: str = "ANTHROPIC_API_KEY",
        max_tokens: int = 1024,
        cost_per_1k_input_usd: float = DEFAULT_STYLE_COST_PER_1K_INPUT_USD,
        cost_per_1k_output_usd: float = DEFAULT_STYLE_COST_PER_1K_OUTPUT_USD,
        responder: Callable[..., dict[str, Any]] | None = None,
    ):
        self.model = model
        self.api_key_env = api_key_env
        self.max_tokens = int(max_tokens)
        self.cost_per_1k_input_usd = float(cost_per_1k_input_usd)
        self.cost_per_1k_output_usd = float(cost_per_1k_output_usd)
        self._responder = responder or _default_style_responder

    def profile(self, *, description: str = "", image_path: str | None = None,
                language: str = "en", feedback: str = "") -> Generation:
        if not (description or "").strip() and not image_path:
            raise ValueError("style profiling needs a description or an image")
        if image_path and not Path(image_path).is_file():
            raise FileNotFoundError(f"style reference image not found: {image_path}")

        start = time.time()
        result = self._responder(
            description=description.strip(),
            image_path=image_path,
            language=language,
            model=self.model,
            max_tokens=self.max_tokens,
            api_key_env=self.api_key_env,
            feedback=feedback,
        )
        style = normalize_style_dict(result.get("style") or {}, description)
        usage = result.get("usage") or {}
        return Generation(
            content=style,
            provider=self.name,
            model=str(result.get("model") or self.model),
            cost_usd=self._cost(usage),
            seconds=round(time.time() - start, 3),
            meta={"input": "style", "saw_image": bool(image_path), "usage": usage},
        )

    def _cost(self, usage: dict[str, Any]) -> float:
        in_tok = float(usage.get("input_tokens") or 0)
        out_tok = float(usage.get("output_tokens") or 0)
        return round(
            in_tok / 1000 * self.cost_per_1k_input_usd
            + out_tok / 1000 * self.cost_per_1k_output_usd,
            6,
        )


def _default_style_responder(
    *,
    description: str,
    image_path: str | None,
    language: str,
    model: str,
    max_tokens: int,
    api_key_env: str,
    feedback: str = "",
) -> dict[str, Any]:
    """Ask Claude for a style guide JSON (paid, live)."""
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} not set - add it to .env (see `cli doctor`).")

    content: list[dict[str, Any]] = []
    if image_path:
        img = Path(image_path)
        data = base64.standard_b64encode(img.read_bytes()).decode("ascii")
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": _image_mime_type(img), "data": data}})
    content.append({"type": "text", "text": style_user_text(description, bool(image_path), language, feedback)})

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=STYLE_SYSTEM,
        messages=[{"role": "user", "content": content}],
    )
    text = next((b.text for b in response.content if getattr(b, "type", None) == "text"), "{}")
    usage = getattr(response, "usage", None)
    return {
        "style": json.loads(text),
        "usage": {
            "input_tokens": getattr(usage, "input_tokens", 0),
            "output_tokens": getattr(usage, "output_tokens", 0),
        },
        "model": getattr(response, "model", model),
    }


def _default_reference_responder(
    *,
    image_paths: list[str],
    prompt: str,
    system: str | None,
    model: str,
    max_tokens: int,
    api_key_env: str,
) -> dict[str, Any]:
    """Send reference image(s) + prompt to Claude and return JSON text (paid, live)."""
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} not set - add it to .env (see `cli doctor`).")

    content: list[dict[str, Any]] = []
    for path in image_paths:
        img = Path(path)
        data = base64.standard_b64encode(img.read_bytes()).decode("ascii")
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": _image_mime_type(img), "data": data}})
    content.append({"type": "text", "text": prompt})

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": content}],
    }
    if system:
        kwargs["system"] = system
    response = client.messages.create(**kwargs)
    text = next((b.text for b in response.content if getattr(b, "type", None) == "text"), "{}")
    usage = getattr(response, "usage", None)
    return {
        "text": text,
        "usage": {
            "input_tokens": getattr(usage, "input_tokens", 0),
            "output_tokens": getattr(usage, "output_tokens", 0),
        },
        "model": getattr(response, "model", model),
    }


def _image_mime_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    if ext == ".webp":
        return "image/webp"
    return "image/png"


def _normalize(verdict: dict[str, Any], clip: Path, prompt: str) -> dict[str, Any]:
    """Turn the model's verdict into the report shape every stage consumes."""
    by_dim = {c.get("dimension"): c for c in verdict.get("checks", []) if isinstance(c, dict)}
    checks = []
    for dim in QC_DIMENSIONS:
        c = by_dim.get(dim, {})
        passed = bool(c.get("passed", True))
        severity = c.get("severity") or ("none" if passed else "high")
        checks.append({
            "dimension": dim,
            "passed": passed,
            "severity": severity,
            "detail": c.get("detail", ""),
            "timestamp": c.get("timestamp"),
        })

    status = qc_report_status(checks)
    return {
        "clip": clip.name,
        "prompt": prompt,
        "checks": checks,
        **status,
        "summary": verdict.get("summary", ""),
    }


def _default_responder(
    *,
    clip_path: str,
    prompt: str,
    model: str,
    max_frames: int,
    max_tokens: int,
    api_key_env: str,
) -> dict[str, Any]:
    """Sample frames and ask Claude for a structured QC verdict (paid, live)."""
    import os

    from ..assembly.ffmpeg_edit import extract_frames

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} not set - add it to .env (see `cli doctor`).")

    with tempfile.TemporaryDirectory() as tmp:
        frames = extract_frames(clip_path, tmp, count=max_frames)
        if not frames:
            raise RuntimeError(
                f"could not sample frames from {clip_path} (need a real mp4 + ffmpeg)."
            )
        content: list[dict[str, Any]] = []
        for frame in frames:
            data = base64.standard_b64encode(Path(frame).read_bytes()).decode("ascii")
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": data},
            })
        content.append({"type": "text", "text": _user_text(prompt)})

        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=_SYSTEM,
            messages=[{"role": "user", "content": content}],
            output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
        )

    text = next((b.text for b in response.content if getattr(b, "type", None) == "text"), "{}")
    verdict = json.loads(text)
    usage = getattr(response, "usage", None)
    return {
        "verdict": verdict,
        "usage": {
            "input_tokens": getattr(usage, "input_tokens", 0),
            "output_tokens": getattr(usage, "output_tokens", 0),
        },
        "model": getattr(response, "model", model),
    }


def _user_text(prompt: str) -> str:
    dims = ", ".join(QC_DIMENSIONS)
    return (
        "These frames are sampled in order from one generated video shot. Review them "
        "against the complete QC context below.\n\n"
        f"{prompt}\n\n"
        f"Review every dimension ({dims}) and return the JSON verdict. Prioritize story "
        "alignment, exact shot instructions, and continuity before cosmetic texture issues. "
        "Low-severity artifacts may be marked failed as warnings, but reserve medium/high "
        "severity for issues worth regenerating. For each dimension, give a one-line "
        "`detail` and a `timestamp` (MM:SS) for the worst frame when relevant."
    )
