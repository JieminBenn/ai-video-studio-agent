"""Cheap real video QC and style profiling via Gemini (invariants #6, #9).

Gemini is the default low-cost M1 QC path: it can inspect the generated MP4 directly,
then returns the same per-dimension report shape as ``FakeVLMCheck`` and
``AnthropicVLMCheck``. Stage code still sees only the ``VLMCheck`` interface, so
switching QC vendors remains a config-only change (invariant #6).

``GeminiStyleProfiler`` extracts a structured style dict from a free-text description
and/or an uploaded reference image via the same Gemini ``generate_content`` API.

Both classes are lazy: importing or constructing them does not require ``google-genai``
or ``GEMINI_API_KEY``; only the live default responder touches the SDK. Unit tests inject
``responder`` and never hit the network.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable

from .base import QC_DIMENSIONS, Generation, ReferenceAnalyzer, StyleProfiler, VLMCheck, qc_report_status
from .openai_compatible import parse_json_content
from .reference_analysis import (
    REFERENCE_ANALYSIS_SYSTEM,
    normalize_reference_analysis,
    reference_analysis_text,
)
from ..style import normalize_style_dict
from .style_prompt import STYLE_SYSTEM, style_user_text

DEFAULT_MODEL = "gemini-2.5-flash-lite"
# Gemini 2.5 Flash-Lite list price: $0.10 / 1M input, $0.40 / 1M output.
DEFAULT_COST_PER_1K_INPUT_USD = 0.0001
DEFAULT_COST_PER_1K_OUTPUT_USD = 0.0004
DEFAULT_STYLE_MODEL = DEFAULT_MODEL
DEFAULT_STYLE_COST_PER_1K_INPUT_USD = DEFAULT_COST_PER_1K_INPUT_USD
DEFAULT_STYLE_COST_PER_1K_OUTPUT_USD = DEFAULT_COST_PER_1K_OUTPUT_USD
DEFAULT_INLINE_VIDEO_MAX_MB = 20.0
_SYSTEM = (
    "You are a strict video QC reviewer for an AI film pipeline. You are given one "
    "generated video shot and the full story-aware QC context it must satisfy. Judge "
    "story and shot intent before surface polish. Mark a dimension as failed only when "
    "there is a clear, defensible problem; cite a timestamp when possible."
)

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
            },
        },
        "summary": {"type": "string"},
    },
    "required": ["checks", "summary"],
}

_MEDIA_RESOLUTION = {
    "low": "MEDIA_RESOLUTION_LOW",
    "medium": "MEDIA_RESOLUTION_MEDIUM",
    "high": "MEDIA_RESOLUTION_HIGH",
    "ultra_high": "MEDIA_RESOLUTION_ULTRA_HIGH",
}


class GeminiVLMCheck(VLMCheck, ReferenceAnalyzer):
    name = "gemini"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        max_tokens: int = 2048,
        media_resolution: str = "low",
        inline_video_max_mb: float = DEFAULT_INLINE_VIDEO_MAX_MB,
        cost_per_1k_input_usd: float = DEFAULT_COST_PER_1K_INPUT_USD,
        cost_per_1k_output_usd: float = DEFAULT_COST_PER_1K_OUTPUT_USD,
        api_key_env: str = "GEMINI_API_KEY",
        max_attempts: int = 3,
        retry_backoff_s: float = 2.0,
        responder: Callable[..., dict[str, Any]] | None = None,
        reference_responder: Callable[..., dict[str, Any]] | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.media_resolution = media_resolution
        self.inline_video_max_mb = inline_video_max_mb
        self.cost_per_1k_input_usd = cost_per_1k_input_usd
        self.cost_per_1k_output_usd = cost_per_1k_output_usd
        self.api_key_env = api_key_env
        self.max_attempts = max(1, int(max_attempts))
        self.retry_backoff_s = max(0.0, float(retry_backoff_s))
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
        usage = _normalized_usage(result.get("usage") or {})
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
        usage = _normalized_usage(result.get("usage") or {})
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
        usage = _normalized_usage(result.get("usage") or {})
        return Generation(
            content=result.get("text", ""),
            provider=self.name,
            model=str(result.get("model") or self.model),
            cost_usd=self._cost(usage),
            seconds=round(time.time() - start, 3),
            meta={"usage": usage, "input": "reference_revise"},
        )

    @property
    def supports_audio_review(self) -> bool:
        return True

    def review(self, clip_path: str, *, prompt: str, **kwargs: Any) -> Generation:
        clip = Path(clip_path)
        if not clip.is_file():
            raise FileNotFoundError(f"VLM QC clip not found: {clip_path}")

        start = time.time()
        result = None
        for attempt in range(self.max_attempts):
            try:
                result = self._responder(
                    clip_path=str(clip),
                    prompt=prompt,
                    model=self.model,
                    max_tokens=self.max_tokens,
                    media_resolution=self.media_resolution,
                    inline_video_max_mb=self.inline_video_max_mb,
                    api_key_env=self.api_key_env,
                )
                break
            except Exception as exc:
                # Transient Gemini server errors (503 high demand, rate limits) are worth
                # retrying; client errors are not and must surface immediately.
                if attempt + 1 >= self.max_attempts or not _is_transient_error(exc):
                    raise
                if self.retry_backoff_s:
                    time.sleep(self.retry_backoff_s * (attempt + 1))
        report = _normalize(result.get("verdict") or {}, clip, prompt)
        usage = _normalized_usage(result.get("usage") or {})
        return Generation(
            content=report,
            provider=self.name,
            model=str(result.get("model") or self.model),
            cost_usd=self._cost(usage),
            seconds=round(time.time() - start, 2),
            meta={
                "usage": usage,
                "media_resolution": self.media_resolution,
                "video_input_mode": result.get("video_input_mode"),
            },
        )

    def _cost(self, usage: dict[str, Any]) -> float:
        in_tok = float(usage.get("input_tokens") or 0)
        out_tok = float(usage.get("output_tokens") or 0)
        return round(
            in_tok / 1000 * self.cost_per_1k_input_usd
            + out_tok / 1000 * self.cost_per_1k_output_usd,
            6,
        )


class GeminiStyleProfiler(StyleProfiler):
    """Profile a free-form style (text and/or one image) into a style dict via Gemini.

    Mirrors ``GeminiVLMCheck``: the live ``generate_content`` call is lazy and lives in an
    injectable responder, so the deterministic core is unit-tested with no SDK or network.
    """

    name = "gemini-style"

    def __init__(
        self,
        model: str = DEFAULT_STYLE_MODEL,
        *,
        api_key_env: str = "GEMINI_API_KEY",
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
        usage = _normalized_usage(result.get("usage") or {})
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
    """Ask Gemini for a style guide JSON (paid, live)."""
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} not set - add it to .env (see `cli doctor`).")
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError(
            "google-genai not installed - run `pip install -e \".[real]\"`."
        ) from exc

    parts = []
    if image_path:
        img = Path(image_path)
        parts.append(types.Part(inline_data=types.Blob(
            data=img.read_bytes(), mime_type=_image_mime_type(img))))
    parts.append(types.Part(text=style_user_text(description, bool(image_path), language, feedback)))

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=types.Content(parts=parts),
        config=types.GenerateContentConfig(
            systemInstruction=STYLE_SYSTEM,
            maxOutputTokens=max_tokens,
            responseMimeType="application/json",
        ),
    )
    text = _response_text(response)
    return {
        "style": parse_json_content(text),
        "usage": _usage_dict(getattr(response, "usage_metadata", None)),
        "model": getattr(response, "model_version", None) or model,
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
    """Send reference image(s) + prompt to Gemini and return JSON text (paid, live)."""
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} not set - add it to .env (see `cli doctor`).")
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError(
            "google-genai not installed - run `pip install -e \".[real]\"`."
        ) from exc

    parts = []
    for path in image_paths:
        img = Path(path)
        parts.append(types.Part(inline_data=types.Blob(
            data=img.read_bytes(), mime_type=_image_mime_type(img))))
    parts.append(types.Part(text=prompt))

    config: dict[str, Any] = {
        "maxOutputTokens": max_tokens,
        "responseMimeType": "application/json",
    }
    if system:
        config["systemInstruction"] = system

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=types.Content(parts=parts),
        config=types.GenerateContentConfig(**config),
    )
    text = _response_text(response)
    return {
        "text": text,
        "usage": _usage_dict(getattr(response, "usage_metadata", None)),
        "model": getattr(response, "model_version", None) or model,
    }


def _image_mime_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    if ext == ".webp":
        return "image/webp"
    return "image/png"


def _normalize(verdict: dict[str, Any], clip: Path, prompt: str) -> dict[str, Any]:
    by_dim = {c.get("dimension"): c for c in verdict.get("checks", []) if isinstance(c, dict)}
    checks = []
    for dim in QC_DIMENSIONS:
        check = by_dim.get(dim)
        missing = check is None
        check = check or {}
        passed = False if missing else bool(check.get("passed", False))
        severity = "high" if missing else check.get("severity") or (
            "none" if passed else "high"
        )
        checks.append({
            "dimension": dim,
            "passed": passed,
            "severity": severity,
            "detail": (
                "required QC dimension missing from provider response"
                if missing else check.get("detail", "")
            ),
            "timestamp": check.get("timestamp"),
        })

    status = qc_report_status(checks)
    return {
        "clip": clip.name,
        "prompt": prompt,
        "checks": checks,
        **status,
        "summary": verdict.get("summary", ""),
    }


_TRANSIENT_STATUS = {429, 500, 502, 503, 504}
_TRANSIENT_MARKERS = ("unavailable", "high demand", "try again", "rate limit", "overloaded", "timeout")


def _is_transient_error(exc: Exception) -> bool:
    """True for transient Gemini API failures worth retrying (5xx, rate limits)."""
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(status, int) and status in _TRANSIENT_STATUS:
        return True
    message = str(exc).lower()
    return any(marker in message for marker in _TRANSIENT_MARKERS)


def _default_responder(
    *,
    clip_path: str,
    prompt: str,
    model: str,
    max_tokens: int,
    media_resolution: str,
    inline_video_max_mb: float,
    api_key_env: str,
) -> dict[str, Any]:
    """Ask Gemini to review a clip directly (paid, live)."""
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} not set - add it to .env (see `cli doctor`).")

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError(
            "google-genai not installed - run `pip install -e \".[real]\"`."
        ) from exc

    clip = Path(clip_path)
    client = genai.Client(api_key=api_key)
    video_part, video_input_mode = _video_part(
        client,
        types,
        clip,
        media_resolution=media_resolution,
        inline_video_max_mb=inline_video_max_mb,
    )
    config_kwargs: dict[str, Any] = {
        "systemInstruction": _SYSTEM,
        "maxOutputTokens": max_tokens,
        "responseMimeType": "application/json",
        "responseSchema": _SCHEMA,
    }
    resolution = _media_resolution(media_resolution, allow_ultra_high=False)
    if resolution and _supports_config_media_resolution(model):
        config_kwargs["mediaResolution"] = resolution

    response = client.models.generate_content(
        model=model,
        contents=types.Content(parts=[video_part, types.Part(text=_user_text(prompt))]),
        config=types.GenerateContentConfig(**config_kwargs),
    )
    text = _response_text(response)
    return {
        "verdict": parse_json_content(text),
        "usage": _usage_dict(getattr(response, "usage_metadata", None)),
        "model": getattr(response, "model_version", None) or model,
        "video_input_mode": video_input_mode,
    }


def _video_part(
    client: Any,
    types: Any,
    clip: Path,
    *,
    media_resolution: str,
    inline_video_max_mb: float,
) -> tuple[Any, str]:
    if clip.stat().st_size <= inline_video_max_mb * 1024 * 1024:
        return (
            types.Part(
                inline_data=types.Blob(data=clip.read_bytes(), mime_type=_mime_type(clip))
            ),
            "inline",
        )

    uploaded = client.files.upload(file=str(clip))
    uploaded = _wait_for_file(client, uploaded)
    return uploaded, "file"


def _wait_for_file(client: Any, file_obj: Any, *, poll_interval_s: float = 2.0) -> Any:
    """Wait for a Files API upload to become usable for video understanding."""
    name = getattr(file_obj, "name", None)
    for _ in range(60):
        state = _state_name(file_obj)
        if state in ("ACTIVE", "FILE_STATE_ACTIVE", "SUCCEEDED"):
            return file_obj
        if state not in ("PROCESSING", "FILE_STATE_PROCESSING", "UNSPECIFIED"):
            raise RuntimeError(f"Gemini file upload failed with state {state}")
        time.sleep(poll_interval_s)
        file_obj = client.files.get(name=name)
    raise RuntimeError("Gemini file upload did not become active before timeout.")


def _state_name(file_obj: Any) -> str:
    state = getattr(file_obj, "state", None)
    if state is None:
        return "ACTIVE"
    return str(getattr(state, "name", state)).split(".")[-1]


def _media_resolution(value: str, *, allow_ultra_high: bool) -> str | None:
    key = (value or "").strip().lower().replace("-", "_")
    if key == "ultra_high" and not allow_ultra_high:
        key = "high"
    return _MEDIA_RESOLUTION.get(key)


def _supports_config_media_resolution(model: str) -> bool:
    """Only send config mediaResolution to model families that advertise it.

    Gemini's video docs currently describe media_resolution as a Gemini 3 control.
    Older Gemini 2.5 models still process video, but sending the field can make the
    API reject the otherwise valid request with a generic INVALID_ARGUMENT.
    """
    normalized = model.removeprefix("models/").lower()
    return normalized.startswith("gemini-3")


def _mime_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".mov":
        return "video/quicktime"
    if ext == ".webm":
        return "video/webm"
    return "video/mp4"


def _response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text:
        return str(text)
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            part_text = getattr(part, "text", None)
            if part_text:
                return str(part_text)
    raise ValueError("Gemini VLM response did not include text JSON.")


def _usage_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return dict(usage)
    out = {}
    for name in (
        "prompt_token_count",
        "candidates_token_count",
        "total_token_count",
        "thoughts_token_count",
    ):
        value = getattr(usage, name, None)
        if value is not None:
            out[name] = value
    return out


def _normalized_usage(usage: dict[str, Any]) -> dict[str, Any]:
    input_tokens = usage.get("input_tokens", usage.get("prompt_token_count", 0))
    output_tokens = usage.get("output_tokens", usage.get("candidates_token_count", 0))
    total_tokens = usage.get("total_tokens", usage.get("total_token_count"))
    normalized = {
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
    }
    if total_tokens is not None:
        normalized["total_tokens"] = int(total_tokens or 0)
    for key, value in usage.items():
        normalized.setdefault(key, value)
    return normalized


def _user_text(prompt: str) -> str:
    dims = ", ".join(QC_DIMENSIONS)
    return (
        "Review this generated video shot against the complete QC context below.\n\n"
        f"{prompt}\n\n"
        f"Check every dimension ({dims}). Prioritize story alignment, exact shot "
        "instructions, and continuity before cosmetic texture issues. Low-severity "
        "artifacts may be marked failed as warnings, but reserve medium/high severity "
        "for issues worth regenerating. Audio policy: verify exact speaker and dialogue, "
        "reject invented speech, confirm synchronized diegetic SFX and natural ambience, "
        "and fail music_absence if any score, song, singing, beat, melody, or underscore "
        "is audible. Return JSON only. For each dimension, provide "
        "a one-line `detail` and include a `timestamp` (MM:SS) for the worst moment "
        "when relevant."
    )
