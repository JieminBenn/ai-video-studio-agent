"""OpenAI-compatible visual QC provider.

Ark/ModelArk, local vLLM-style vision servers, and several hosted gateways expose
vision models through the chat-completions request shape: a user message can mix
``image_url`` parts and text. This adapter samples a generated clip into ordered
frames, sends those frames to a selected vision-capable model, and normalizes the
JSON verdict into the same QC report shape used by the rest of the pipeline.

The live network call is lazy. Constructing the provider needs no API key; only a
real ``review()`` call with the default transport reads the key and sends HTTP.
Unit tests inject the frame sampler and transport so they never touch ffmpeg,
network, or paid providers.
"""

from __future__ import annotations

import base64
import http.client
import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from ..costing import CostEstimate, estimate_token_cost
from .base import (
    QC_DIMENSIONS,
    Generation,
    ReferenceAnalyzer,
    StyleProfiler,
    VLMCheck,
    qc_report_status,
)
from .openai_compatible import parse_json_content
from .style_prompt import STYLE_SYSTEM, style_user_text

Transport = Callable[[str, dict[str, Any], dict[str, str], float], dict[str, Any]]


def _send_with_reset_retry(
    transport: Transport,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_s: float,
    *,
    max_retries: int,
    backoff_s: float,
    label: str,
) -> dict[str, Any]:
    """Call ``transport``, retrying transient connection resets with backoff.

    A dropped/reset connection surfaces as ``http.client.RemoteDisconnected`` — a
    ``ConnectionResetError`` (not a ``URLError``) — or a truncated ``IncompleteRead``. Either is
    transient, so retry it rather than aborting a paid stage with the raw
    "Remote end closed connection without response". Other errors (a wrapped HTTP failure, a
    read timeout) propagate unchanged."""
    last: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return transport(url, payload, headers, timeout_s)
        except (ConnectionError, http.client.IncompleteRead) as exc:
            last = exc
            if attempt < max_retries:
                time.sleep(backoff_s * attempt)
                continue
            raise RuntimeError(
                f"{label} request failed after {max_retries} attempts: {exc}"
            ) from exc
    raise RuntimeError(f"{label} request failed: {last}")
FrameSampler = Callable[[str, str, int], list[str]]

DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_MODEL = "doubao-seed-2-0-lite-260428"
DEFAULT_COST_PER_1K_INPUT_USD = 0.0
DEFAULT_COST_PER_1K_OUTPUT_USD = 0.0

_SYSTEM = (
    "You are a strict visual QC reviewer for an AI film pipeline. You are shown "
    "ordered frames sampled from one generated video shot plus the full story-aware "
    "QC context it must satisfy. Judge story and shot intent before surface polish. "
    "Mark a dimension as failed only when there is a clear, defensible problem; cite "
    "a frame/timestamp when possible."
)

_REFERENCE_SYSTEM = (
    "You classify uploaded visual references for an AI filmmaking pipeline. Inspect "
    "the image itself and decide whether it primarily represents a character, a "
    "location/background, or a global visual style. Return strict JSON only."
)

_REFERENCE_TASK_SYSTEM = (
    "You inspect uploaded visual references for an AI filmmaking pipeline. "
    "Follow the user's exact task and output schema. Treat the images as authoritative "
    "visual evidence. This is not video QC; do not return a QC verdict unless explicitly asked."
)


class OpenAICompatibleVLMCheck(VLMCheck, ReferenceAnalyzer):
    """Frame-based video QC through an OpenAI-compatible vision chat API."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        provider_name: str = "openai-compatible-vlm",
        api_key_env: str | None = None,
        api_key: str | None = None,
        max_frames: int = 4,
        max_tokens: int = 2048,
        detail: str | None = None,
        temperature: float = 0.0,
        timeout_s: float = 120.0,
        max_retries: int = 3,
        retry_backoff_s: float = 1.5,
        json_response_format: bool = False,
        cost_per_1k_input_usd: float = DEFAULT_COST_PER_1K_INPUT_USD,
        cost_per_1k_output_usd: float = DEFAULT_COST_PER_1K_OUTPUT_USD,
        cost_per_million_input: float | None = None,
        cost_per_million_cached_input: float | None = None,
        cost_per_million_output: float | None = None,
        cost_currency: str = "USD",
        usd_per_native_unit: float = 1.0,
        pricing_source: str = "config",
        pricing_as_of: str = "",
        transport: Transport | None = None,
        frame_sampler: FrameSampler | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = provider_name
        self.api_key_env = api_key_env
        self.api_key = api_key
        self.max_frames = max(1, int(max_frames))
        self.max_tokens = int(max_tokens)
        self.detail = detail
        self.temperature = float(temperature)
        self.timeout_s = float(timeout_s)
        self.max_retries = max(1, int(max_retries))
        self.retry_backoff_s = max(0.0, float(retry_backoff_s))
        self.json_response_format = bool(json_response_format)
        self.cost_per_1k_input_usd = float(cost_per_1k_input_usd)
        self.cost_per_1k_output_usd = float(cost_per_1k_output_usd)
        self.cost_per_million_input = float(
            self.cost_per_1k_input_usd * 1000
            if cost_per_million_input is None
            else cost_per_million_input
        )
        self.cost_per_million_cached_input = float(
            self.cost_per_million_input
            if cost_per_million_cached_input is None
            else cost_per_million_cached_input
        )
        self.cost_per_million_output = float(
            self.cost_per_1k_output_usd * 1000
            if cost_per_million_output is None
            else cost_per_million_output
        )
        self.cost_currency = str(cost_currency or "USD").upper()
        self.usd_per_native_unit = float(usd_per_native_unit)
        self.pricing_source = str(pricing_source or "config")
        self.pricing_as_of = str(pricing_as_of or "")
        self._raw_transport = transport or _urlopen_transport
        self._transport = self._retrying_transport
        self._frame_sampler = frame_sampler or _default_frame_sampler

    def _retrying_transport(
        self, url: str, payload: dict[str, Any], headers: dict[str, str], timeout_s: float
    ) -> dict[str, Any]:
        return _send_with_reset_retry(
            self._raw_transport, url, payload, headers, timeout_s,
            max_retries=self.max_retries, backoff_s=self.retry_backoff_s, label="VLM",
        )

    def review(self, clip_path: str, *, prompt: str, **kwargs: Any) -> Generation:
        clip = Path(clip_path)
        if not clip.is_file():
            raise FileNotFoundError(f"VLM QC clip not found: {clip_path}")

        start = time.time()
        with tempfile.TemporaryDirectory() as tmp:
            frames = self._frame_sampler(str(clip), tmp, self.max_frames)
            if not frames:
                raise RuntimeError(
                    f"could not sample frames from {clip_path} (need a real mp4 + ffmpeg)."
                )
            payload = self._payload(prompt, frames)
            data = self._transport(
                f"{self.base_url}/chat/completions",
                payload,
                self._headers(),
                self.timeout_s,
            )

        text = _message_content(data)
        verdict = parse_json_content(text)
        report = _normalize(verdict if isinstance(verdict, dict) else {}, clip, prompt)
        usage = _normalized_usage(data.get("usage") or {})
        estimate = self._estimate(usage, str(data.get("model") or self.model))
        return Generation(
            content=report,
            provider=self.name,
            model=str(data.get("model") or self.model),
            cost_usd=estimate.cost_usd,
            seconds=round(time.time() - start, 3),
            meta={
                "base_url": self.base_url,
                "usage": usage,
                "cost_tracking": estimate.tracking,
                "frames_sampled": len(frames),
                "finish_reason": _finish_reason(data),
                "detail": self.detail,
            },
        )

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
        payload = self._reference_payload(image, aliases or [], user_note)
        data = self._transport(
            f"{self.base_url}/chat/completions",
            payload,
            self._headers(),
            self.timeout_s,
        )
        analysis = _normalize_reference_analysis(parse_json_content(_message_content(data)))
        usage = _normalized_usage(data.get("usage") or {})
        estimate = self._estimate(usage, str(data.get("model") or self.model))
        return Generation(
            content=analysis,
            provider=self.name,
            model=str(data.get("model") or self.model),
            cost_usd=estimate.cost_usd,
            seconds=round(time.time() - start, 3),
            meta={
                "base_url": self.base_url,
                "usage": usage,
                "cost_tracking": estimate.tracking,
                "detail": self.detail,
                "input": "reference_image",
            },
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
        payload = self._reference_task_payload(
            prompt,
            images[: self.max_frames],
            json_response=self.json_response_format,
        )
        data = self._transport(
            f"{self.base_url}/chat/completions",
            payload,
            self._headers(),
            self.timeout_s,
        )
        content = parse_json_content(_message_content(data))
        usage = _normalized_usage(data.get("usage") or {})
        estimate = self._estimate(usage, str(data.get("model") or self.model))
        return Generation(
            content=content if isinstance(content, dict) else {},
            provider=self.name,
            model=str(data.get("model") or self.model),
            cost_usd=estimate.cost_usd,
            seconds=round(time.time() - start, 3),
            meta={
                "base_url": self.base_url,
                "usage": usage,
                "cost_tracking": estimate.tracking,
                "detail": self.detail,
                "input": "reference_describe",
            },
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
        payload = self._reference_task_payload(
            prompt,
            images[: self.max_frames],
            json_response=False,
        )
        data = self._transport(
            f"{self.base_url}/chat/completions",
            payload,
            self._headers(),
            self.timeout_s,
        )
        usage = _normalized_usage(data.get("usage") or {})
        estimate = self._estimate(usage, str(data.get("model") or self.model))
        return Generation(
            content=_message_content(data),
            provider=self.name,
            model=str(data.get("model") or self.model),
            cost_usd=estimate.cost_usd,
            seconds=round(time.time() - start, 3),
            meta={
                "base_url": self.base_url,
                "usage": usage,
                "cost_tracking": estimate.tracking,
                "input": "reference_revise",
            },
        )

    def _payload(self, prompt: str, frames: list[str]) -> dict[str, Any]:
        content = []
        for frame in frames:
            image_url: dict[str, Any] = {"url": _data_url(Path(frame))}
            if self.detail:
                image_url["detail"] = self.detail
            content.append({"type": "image_url", "image_url": image_url})
        content.append({"type": "text", "text": _user_text(prompt)})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": content},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.json_response_format:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _reference_task_payload(
        self,
        prompt: str,
        images: list[str],
        *,
        json_response: bool,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        for image in images:
            image_url: dict[str, Any] = {"url": _data_url(Path(image))}
            if self.detail:
                image_url["detail"] = self.detail
            content.append({"type": "image_url", "image_url": image_url})
        content.append({"type": "text", "text": prompt})
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _REFERENCE_TASK_SYSTEM},
                {"role": "user", "content": content},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if json_response:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _reference_payload(
        self,
        image: Path,
        aliases: list[str],
        user_note: str,
    ) -> dict[str, Any]:
        image_url: dict[str, Any] = {"url": _data_url(image)}
        if self.detail:
            image_url["detail"] = self.detail
        text = (
            "Analyze this one uploaded reference independently. "
            f"Its aliases are: {', '.join(aliases) or '(none)'}. "
            f"The user's note is: {user_note or '(none)'}. "
            "Return an object with target_type (character, location, style, or unknown), "
            "target_id (a concise character/location identifier, or global for style), "
            "confidence (0 to 1), reason, and visual_summary. Do not infer from upload "
            "order alone; use the visible content and the user's note."
        )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _REFERENCE_SYSTEM},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": image_url},
                    {"type": "text", "text": text},
                ]},
            ],
            "temperature": 0.0,
            "max_tokens": min(self.max_tokens, 1024),
        }
        if self.json_response_format:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = self.api_key or (os.getenv(self.api_key_env) if self.api_key_env else None)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        elif self.api_key_env:
            raise RuntimeError(f"{self.api_key_env} not set - add it to .env.")
        return headers

    def _estimate(self, usage: dict[str, Any], model: str) -> CostEstimate:
        return estimate_token_cost(
            usage,
            input_per_million=self.cost_per_million_input,
            cached_input_per_million=self.cost_per_million_cached_input,
            output_per_million=self.cost_per_million_output,
            native_currency=self.cost_currency,
            usd_per_native_unit=self.usd_per_native_unit,
            model=model,
            pricing_source=self.pricing_source,
            pricing_as_of=self.pricing_as_of,
        )


class OpenAICompatibleStyleProfiler(StyleProfiler):
    """Profile a free-form style (text and/or one image) into a reusable style dict.

    Uses the same OpenAI-compatible chat-completions vision shape as the VLM QC adapter,
    so one hosted/local vision model serves both. The live call is lazy; tests inject a
    transport and never touch the network.
    """

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        provider_name: str = "openai-compatible-style",
        api_key_env: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 1024,
        detail: str | None = None,
        temperature: float = 0.2,
        timeout_s: float = 120.0,
        max_retries: int = 3,
        retry_backoff_s: float = 1.5,
        json_response_format: bool = False,
        cost_per_1k_input_usd: float = DEFAULT_COST_PER_1K_INPUT_USD,
        cost_per_1k_output_usd: float = DEFAULT_COST_PER_1K_OUTPUT_USD,
        transport: Transport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = provider_name
        self.api_key_env = api_key_env
        self.api_key = api_key
        self.max_tokens = int(max_tokens)
        self.detail = detail
        self.temperature = float(temperature)
        self.timeout_s = float(timeout_s)
        self.max_retries = max(1, int(max_retries))
        self.retry_backoff_s = max(0.0, float(retry_backoff_s))
        self.json_response_format = bool(json_response_format)
        self.cost_per_1k_input_usd = float(cost_per_1k_input_usd)
        self.cost_per_1k_output_usd = float(cost_per_1k_output_usd)
        self._raw_transport = transport or _urlopen_transport
        self._transport = self._retrying_transport

    def _retrying_transport(
        self, url: str, payload: dict[str, Any], headers: dict[str, str], timeout_s: float
    ) -> dict[str, Any]:
        return _send_with_reset_retry(
            self._raw_transport, url, payload, headers, timeout_s,
            max_retries=self.max_retries, backoff_s=self.retry_backoff_s, label="Style profiler",
        )

    def profile(
        self,
        *,
        description: str = "",
        image_path: str | None = None,
        language: str = "en",
        feedback: str = "",
    ) -> Generation:
        if not (description or "").strip() and not image_path:
            raise ValueError("style profiling needs a description or an image")
        if image_path and not Path(image_path).is_file():
            raise FileNotFoundError(f"style reference image not found: {image_path}")

        start = time.time()
        payload = self._payload(description.strip(), image_path, language, feedback)
        data = self._transport(
            f"{self.base_url}/chat/completions",
            payload,
            self._headers(),
            self.timeout_s,
        )
        style = _normalize_style(parse_json_content(_message_content(data)), description)
        usage = _normalized_usage(data.get("usage") or {})
        return Generation(
            content=style,
            provider=self.name,
            model=str(data.get("model") or self.model),
            cost_usd=self._cost(usage),
            seconds=round(time.time() - start, 3),
            meta={"base_url": self.base_url, "usage": usage, "input": "style",
                  "saw_image": bool(image_path)},
        )

    def _payload(self, description: str, image_path: str | None, language: str, feedback: str = "") -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        if image_path:
            image_url: dict[str, Any] = {"url": _data_url(Path(image_path))}
            if self.detail:
                image_url["detail"] = self.detail
            content.append({"type": "image_url", "image_url": image_url})
        content.append({"type": "text", "text": style_user_text(description, bool(image_path), language, feedback)})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": STYLE_SYSTEM},
                {"role": "user", "content": content},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.json_response_format:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = self.api_key or (os.getenv(self.api_key_env) if self.api_key_env else None)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        elif self.api_key_env:
            raise RuntimeError(f"{self.api_key_env} not set - add it to .env.")
        return headers

    def _cost(self, usage: dict[str, Any]) -> float:
        in_tok = float(usage.get("input_tokens") or 0)
        out_tok = float(usage.get("output_tokens") or 0)
        return round(
            in_tok / 1000 * self.cost_per_1k_input_usd
            + out_tok / 1000 * self.cost_per_1k_output_usd,
            6,
        )


def _normalize_style(value: Any, description: str = "") -> dict[str, Any]:
    from ..style import normalize_style_dict

    return normalize_style_dict(value, description)


def _normalize(verdict: dict[str, Any], clip: Path, prompt: str) -> dict[str, Any]:
    by_dim = {c.get("dimension"): c for c in verdict.get("checks", []) if isinstance(c, dict)}
    checks = []
    for dim in QC_DIMENSIONS:
        check = by_dim.get(dim, {})
        passed = bool(check.get("passed", True))
        severity = check.get("severity") or ("none" if passed else "high")
        checks.append({
            "dimension": dim,
            "passed": passed,
            "severity": severity,
            "detail": check.get("detail", ""),
            "timestamp": check.get("timestamp"),
        })

    return {
        "clip": clip.name,
        "prompt": prompt,
        "checks": checks,
        **qc_report_status(checks),
        "summary": verdict.get("summary", ""),
    }


def _normalize_reference_analysis(value: Any) -> dict[str, Any]:
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


def _default_frame_sampler(clip_path: str, out_dir: str, count: int) -> list[str]:
    from ..assembly.ffmpeg_edit import extract_frames

    return extract_frames(clip_path, out_dir, count=count)


def _data_url(path: Path) -> str:
    data = base64.standard_b64encode(path.read_bytes()).decode("ascii")
    return f"data:{_mime_type(path)};base64,{data}"


def _mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "image/png"


def _urlopen_transport(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_s: float,
) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"VLM request failed with HTTP {exc.code}: {body}") from exc


def _message_content(data: dict[str, Any]) -> str:
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("OpenAI-compatible VLM response missing choices[0].message.content") from exc
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content)


def _finish_reason(data: dict[str, Any]) -> str | None:
    try:
        return data["choices"][0].get("finish_reason")
    except (KeyError, IndexError, TypeError):
        return None


def _normalized_usage(usage: dict[str, Any]) -> dict[str, Any]:
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
    total_tokens = usage.get("total_tokens")
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
        "These frames are sampled in order from one generated video shot. Review them "
        "against the complete QC context below.\n\n"
        f"{prompt}\n\n"
        f"Review every dimension ({dims}) and return JSON only with this shape: "
        '{"summary": string, "checks": [{"dimension": one of the listed dimensions, '
        '"passed": boolean, "severity": "none"|"low"|"medium"|"high", '
        '"detail": string, "timestamp": string|null}]}. Prioritize story alignment, '
        "exact shot instructions, identity continuity, and physically plausible motion "
        "before cosmetic texture issues. Low-severity artifacts may be warnings; use "
        "medium/high severity for issues worth regenerating."
    )
