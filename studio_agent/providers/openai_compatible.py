"""OpenAI-compatible LLM provider.

Many hosted and open-weight runtimes expose the OpenAI chat-completions shape:
OpenAI-compatible gateways, vLLM/SGLang workers, llama.cpp server, and Ollama's
compatibility endpoint. This adapter lets Studio Agent swap those in through
``config.yaml`` without changing stage code.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from .base import Generation, LLM
from ..costing import estimate_token_cost

Transport = Callable[[str, dict[str, Any], dict[str, str], float], dict[str, Any]]


class OpenAICompatibleLLM(LLM):
    name = "openai-compatible"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key_env: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.4,
        timeout_s: float = 120.0,
        stream: bool = True,
        max_json_attempts: int = 2,
        json_response_format: bool = True,
        max_retries: int = 3,
        retry_backoff_s: float = 1.5,
        cost_per_million_input: float = 0.0,
        cost_per_million_cached_input: float | None = None,
        cost_per_million_output: float = 0.0,
        cost_currency: str = "USD",
        usd_per_native_unit: float = 1.0,
        pricing_source: str = "config",
        pricing_as_of: str = "",
        transport: Transport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.api_key = api_key
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.stream = bool(stream)
        self.max_json_attempts = max(1, int(max_json_attempts))
        self.json_response_format = bool(json_response_format)
        self.max_retries = max(1, int(max_retries))
        self.retry_backoff_s = max(0.0, float(retry_backoff_s))
        self.cost_per_million_input = float(cost_per_million_input)
        self.cost_per_million_cached_input = float(
            cost_per_million_input
            if cost_per_million_cached_input is None
            else cost_per_million_cached_input
        )
        self.cost_per_million_output = float(cost_per_million_output)
        self.cost_currency = str(cost_currency or "USD").upper()
        self.usd_per_native_unit = float(usd_per_native_unit)
        self.pricing_source = str(pricing_source or "config")
        self.pricing_as_of = str(pricing_as_of or "")
        if transport is not None:
            self._transport = transport
        elif self.stream:
            self._transport = _urlopen_stream_transport
        else:
            self._transport = _urlopen_transport

    def complete(self, prompt: str, *, system: str | None = None) -> Generation:
        return self._chat(prompt, system=system)

    def complete_json(self, prompt: str, *, system: str | None = None) -> Generation:
        last_error = None
        json_prompt = prompt
        for attempt in range(1, self.max_json_attempts + 1):
            gen = self._chat(
                json_prompt,
                system=system,
                response_format=(
                    {"type": "json_object"} if self.json_response_format else None
                ),
            )
            try:
                parsed = parse_json_content(gen.content)
            except ValueError as exc:
                last_error = exc
                json_prompt = (
                    f"{prompt}\n\nPrevious JSON parse error: {exc}\n"
                    "Return the same answer again as complete, valid JSON only. "
                    "Do not include markdown, commentary, trailing prose, comments, "
                    "or multiline string values; use escaped \\n inside single-line "
                    "JSON strings when needed."
                )
                continue
            gen.content = parsed
            gen.meta["json_attempts"] = attempt
            return gen
        raise ValueError(
            "LLM did not return valid JSON after "
            f"{self.max_json_attempts} attempt(s): {last_error}"
        )

    def _chat(
        self,
        prompt: str,
        *,
        system: str | None = None,
        response_format: dict[str, str] | None = None,
    ) -> Generation:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if response_format:
            payload["response_format"] = response_format
        if self.stream:
            # Stream so tokens (incl. a reasoning model's thinking tokens) arrive continuously.
            # The socket read timeout then acts as a per-chunk *stall* guard rather than a
            # whole-generation deadline, so a slow-but-progressing model (e.g. deepseek-v4-pro
            # on a rich 中文 storyboard scene) no longer trips a read timeout mid-generation.
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}

        headers = {"Content-Type": "application/json"}
        key = self.api_key or (os.getenv(self.api_key_env) if self.api_key_env else None)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        elif self.api_key_env:
            raise RuntimeError(f"{self.api_key_env} not set - add it to .env (see `cli doctor`).")

        start = time.time()
        data = self._request_with_retry(
            f"{self.base_url}/chat/completions", payload, headers
        )
        content = _message_content(data)
        usage = data.get("usage") or {}
        estimate = estimate_token_cost(
            usage,
            input_per_million=self.cost_per_million_input,
            cached_input_per_million=self.cost_per_million_cached_input,
            output_per_million=self.cost_per_million_output,
            native_currency=self.cost_currency,
            usd_per_native_unit=self.usd_per_native_unit,
            model=str(data.get("model") or self.model),
            pricing_source=self.pricing_source,
            pricing_as_of=self.pricing_as_of,
        )
        return Generation(
            content=content,
            provider=self.name,
            model=str(data.get("model") or self.model),
            cost_usd=estimate.cost_usd,
            seconds=round(time.time() - start, 3),
            meta={
                "base_url": self.base_url,
                "usage": usage,
                "cost_tracking": estimate.tracking,
                "finish_reason": _finish_reason(data),
            },
        )

    def _request_with_retry(
        self, url: str, payload: dict[str, Any], headers: dict[str, str]
    ) -> dict[str, Any]:
        """Call the transport, retrying transient connection blips with backoff.

        Mirrors the video Ark path (``providers/volcengine.py``): a connection-phase failure
        (DNS miss, dropped/refused connection, connect timeout) or a 5xx is transient and worth
        retrying; a 4xx (auth/bad request) is terminal; and a read-phase ``TimeoutError`` ("The
        read operation timed out") means the stream stalled (no token for ``timeout_s``) or, for a
        non-streaming profile, the whole generation ran long — either way retrying re-sends the
        whole prompt and will likely time out again, so it is terminal with an actionable message.
        """
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._transport(url, payload, headers, self.timeout_s)
            except TimeoutError as exc:
                raise RuntimeError(
                    f"LLM read timed out after {self.timeout_s:.0f}s ({self.model}). "
                    "Raise llm_timeout_s for this profile or choose a faster model."
                ) from exc
            except urllib.error.HTTPError as exc:
                if exc.code >= 500 and attempt < self.max_retries:
                    last_exc = exc
                    time.sleep(self.retry_backoff_s * attempt)
                    continue
                body = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"LLM request failed with HTTP {exc.code}: {body}"
                ) from exc
            except urllib.error.URLError as exc:
                if _is_transient_urlerror(exc) and attempt < self.max_retries:
                    last_exc = exc
                    time.sleep(self.retry_backoff_s * attempt)
                    continue
                raise RuntimeError(
                    f"LLM request failed: {getattr(exc, 'reason', exc)}"
                ) from exc
            except (ConnectionError, http.client.IncompleteRead) as exc:
                # A dropped/reset connection or a truncated read is transient. Crucially,
                # ``http.client.RemoteDisconnected`` ("Remote end closed connection without
                # response") is a ``ConnectionResetError`` — NOT a ``URLError`` — so it slips
                # past the branch above; retry it with backoff instead of aborting the stage.
                if attempt < self.max_retries:
                    last_exc = exc
                    time.sleep(self.retry_backoff_s * attempt)
                    continue
                raise RuntimeError(f"LLM request failed: {exc}") from exc
        raise RuntimeError(f"LLM request failed after {self.max_retries} attempts: {last_exc}")


def _is_transient_urlerror(exc: urllib.error.URLError) -> bool:
    """A connection-phase failure worth retrying: a DNS lookup miss, a dropped/refused
    connection, or a connect-phase socket timeout (which urllib wraps as ``exc.reason``)."""
    reason = getattr(exc, "reason", None)
    return isinstance(reason, (socket.gaierror, ConnectionError, TimeoutError))


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
    # HTTPError / URLError / TimeoutError propagate to _request_with_retry, which classifies
    # them (retry transient, terminate on 4xx / read timeout) — see its docstring.
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _urlopen_stream_transport(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_s: float,
) -> dict[str, Any]:
    """Stream a chat completion and reassemble it into the non-streaming response shape.

    ``timeout_s`` is set on the underlying socket, so it applies to *each* read of the SSE
    body rather than to the whole generation: as long as the server keeps sending chunks
    within ``timeout_s`` of one another the read never times out, while a genuinely stalled
    connection still raises ``TimeoutError`` (classified as terminal by ``_request_with_retry``).
    """
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        # Iterating the response yields one line per read, each bounded by the socket timeout.
        return _consume_sse_stream(resp, model=payload.get("model"))


def _consume_sse_stream(lines: Any, *, model: str | None = None) -> dict[str, Any]:
    """Fold an OpenAI-compatible ``text/event-stream`` body into a single response dict.

    Accumulates ``choices[0].delta.content`` (a reasoning model's ``reasoning_content`` deltas
    are intentionally ignored — they keep the socket alive but are not part of the answer),
    the final ``finish_reason``, and the trailing usage-only chunk emitted by
    ``stream_options.include_usage``.
    """
    content_parts: list[str] = []
    finish_reason: str | None = None
    usage: dict[str, Any] = {}
    for raw in lines:
        line = (
            raw.decode("utf-8", errors="replace")
            if isinstance(raw, (bytes, bytearray))
            else str(raw)
        ).strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        if chunk.get("model"):
            model = chunk["model"]
        if chunk.get("usage"):
            usage = chunk["usage"]
        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta") or {}
        piece = delta.get("content")
        if piece:
            content_parts.append(str(piece))
        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]
    return {
        "choices": [
            {"message": {"content": "".join(content_parts)}, "finish_reason": finish_reason}
        ],
        "usage": usage,
        "model": model,
    }


def _message_content(data: dict[str, Any]) -> str:
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("OpenAI-compatible response missing choices[0].message.content") from exc
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


def parse_json_content(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        raise ValueError("empty response")
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))

    start_candidates = [idx for idx in (stripped.find("{"), stripped.find("[")) if idx != -1]
    if not start_candidates:
        raise ValueError("response does not contain JSON")
    start = min(start_candidates)
    decoder = json.JSONDecoder()
    try:
        parsed, _ = decoder.raw_decode(stripped[start:])
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON response: {exc}") from exc
    return parsed
