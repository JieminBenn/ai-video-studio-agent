"""Ark Seedance video providers.

This provider calls native Ark video generation APIs directly, avoiding aggregator
pricing. It is lazy like the other real providers: importing or constructing it needs
no network and no key. Live generation requires the configured API key env var.

Reference docs verified 2026-06-16:
- https://www.volcengine.com/docs/82379/1520757?lang=zh
- https://www.volcengine.com/docs/82379/1521309?lang=zh
- https://www.volcengine.com/docs/82379/1298459?lang=zh
- https://www.volcengine.com/docs/82379/1544106?lang=zh
- https://docs.byteplus.com/en/docs/ModelArk/1330626
- https://docs.byteplus.com/en/docs/ModelArk/1520757
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import shutil
import socket
import time
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .base import Generation, VideoCapabilities, VideoGen

DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_MODEL = "doubao-seedance-2-0-fast-260128"
BYTEPLUS_BASE_URL = "https://ark.ap-southeast.bytepluses.com/api/v3"
BYTEPLUS_DEFAULT_MODEL = "dreamina-seedance-2-0-fast-260128"
DEFAULT_RESOLUTION = "720p"
DEFAULT_ASPECT_RATIO = "16:9"
DEFAULT_COST_PER_MILLION_TOKENS_CNY = 37.0
DEFAULT_CNY_TO_USD = 0.14
BYTEPLUS_DEFAULT_COST_PER_1K_TOKENS_USD = 0.0056
MAX_IMAGE_INPUTS = 9
TERMINAL_FAILURE_STATUSES = {"failed", "expired", "cancelled"}
PENDING_STATUSES = {"queued", "running"}
# A paid task is created (and charged) up front, then polled up to ~180 times. A single
# transient network blip — a DNS hiccup or a dropped connection — must not abort the whole
# run, so connection-phase failures and 5xx responses are retried a few times with backoff.
# Client (4xx) errors and read timeouts stay terminal (no point retrying auth or a slow render).
MAX_NETWORK_RETRIES = 2
RETRY_BACKOFF_S = 0.5

Downloader = Callable[[str, str], None]


class ArkSeedanceVideoGen(VideoGen):
    name = "volcengine"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        provider_name: str = "volcengine",
        provider_label: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        resolution: str = DEFAULT_RESOLUTION,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        generate_audio: bool = False,
        return_last_frame: bool = True,
        api_key_env: str = "ARK_API_KEY",
        max_image_inputs: int = MAX_IMAGE_INPUTS,
        poll_interval_s: float = 5.0,
        max_polls: int = 180,
        timeout_s: float = 60.0,
        cost_per_million_tokens_cny: float | None = DEFAULT_COST_PER_MILLION_TOKENS_CNY,
        cny_to_usd: float = DEFAULT_CNY_TO_USD,
        cost_per_1k_tokens_usd: float | None = None,
        pricing_source: str = "config",
        pricing_as_of: str = "",
        include_reference_images: bool = True,
        include_last_frame_ref: bool = True,
        requester: Callable[..., dict[str, Any]] | None = None,
        downloader: Downloader | None = None,
    ):
        self.name = provider_name
        self.provider_label = provider_label or provider_name
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.resolution = resolution
        self.aspect_ratio = aspect_ratio
        self.generate_audio = generate_audio
        self.return_last_frame = return_last_frame
        self.api_key_env = api_key_env
        self.max_image_inputs = max_image_inputs
        self.poll_interval_s = poll_interval_s
        self.max_polls = max_polls
        self.timeout_s = timeout_s
        self.cost_per_million_tokens_cny = cost_per_million_tokens_cny
        self.cny_to_usd = cny_to_usd
        self.cost_per_1k_tokens_usd = cost_per_1k_tokens_usd
        self.pricing_source = str(pricing_source or "config")
        self.pricing_as_of = str(pricing_as_of or "")
        self.include_reference_images = include_reference_images
        self.include_last_frame_ref = include_last_frame_ref
        self._requester = requester or _request_json
        self._downloader = downloader or _download_url

    @property
    def capabilities(self) -> VideoCapabilities:
        # This adapter always sends the mandatory storyboard keyframe alone because Ark
        # rejects mixing first/last-frame content with reference media. Report that actual
        # behavior so stage prompts never promise inputs ``generate`` will discard.
        return VideoCapabilities(
            supports_reference_images=False,
            supports_last_frame=False,
            supports_native_audio=(
                self.generate_audio and _supports_native_audio(self.model)
            ),
            max_image_inputs=self.max_image_inputs,
            supports_storyboard_grid=getattr(self, "_supports_storyboard_grid", False),
            min_duration_s=4,
            max_duration_s=15,
        )

    def generate(self, prompt: str, *, out_path: str, **kwargs: Any) -> Generation:
        keyframe_path = kwargs.get("keyframe_path")
        if not keyframe_path:
            raise ValueError(
                f"{self.provider_label} Seedance video generation requires keyframe_path."
            )

        requested_reference_images = list(kwargs.get("reference_images") or [])
        requested_last_frame_ref = kwargs.get("last_frame_ref")
        # Ark rejects any request that mixes first/last-frame content with reference
        # media (HTTP 400 "first/last frame content cannot be mixed with reference media
        # content"). The keyframe is mandatory and is always sent as the first frame —
        # the clip must match the storyboard image, which was already
        # reference-conditioned at the storyboard stage — so reference media (extra
        # reference images and the carried previous last frame) is always dropped here
        # to keep the request valid. The ``include_*`` flags remain for config/capability
        # reporting but cannot re-enable mixing.
        reference_images: list[str] = []
        last_frame_ref: str | None = None
        image_inputs = _image_inputs(keyframe_path, reference_images, last_frame_ref)
        _guard_inputs(
            [path for path, _role in image_inputs],
            self.max_image_inputs,
            model=self.model,
            provider_label=self.provider_label,
        )

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"{self.api_key_env} not set - add it to .env (see `cli doctor`).")

        duration = _seedance_duration(kwargs.get("duration_s", 4.0))
        ratio = kwargs.get("aspect_ratio") or kwargs.get("ratio") or self.aspect_ratio
        resolution = kwargs.get("resolution") or self.resolution
        generate_audio = bool(kwargs.get("generate_audio", self.generate_audio))

        body = {
            "model": self.model,
            "content": [
                {"type": "text", "text": prompt.strip()},
                *[
                    {
                        "type": "image_url",
                        "image_url": {"url": _data_url(path)},
                        "role": role,
                    }
                    for path, role in image_inputs
                ],
            ],
            "resolution": resolution,
            "ratio": ratio,
            "duration": duration,
            "generate_audio": generate_audio,
            "watermark": False,
            "return_last_frame": bool(kwargs.get("return_last_frame", self.return_last_frame)),
        }
        if _supports_seed(self.model) and kwargs.get("seed") is not None:
            body["seed"] = int(kwargs["seed"])

        start = time.time()
        try:
            create = self._requester(
                "POST",
                f"{self.base_url}/contents/generations/tasks",
                api_key=api_key,
                json_body=body,
                timeout_s=self.timeout_s,
            )
        except RuntimeError as exc:
            friendly = _friendly_input_moderation_error(exc, self.provider_label)
            if friendly is exc:
                raise
            raise friendly from exc
        task_id = create.get("id") if isinstance(create, dict) else None
        if not task_id:
            raise RuntimeError(f"{self.provider_label} create task response did not include a task id.")

        task = self._poll_task(str(task_id), api_key)
        video_url = _video_url(task)
        self._downloader(video_url, out_path)
        elapsed = round(time.time() - start, 2)

        usage = task.get("usage") if isinstance(task.get("usage"), dict) else {}
        cost_cny = 0.0
        if self.cost_per_1k_tokens_usd is None:
            cost_cny = _estimated_cost_cny(
                usage,
                cost_per_million_tokens_cny=self.cost_per_million_tokens_cny,
            )
        cost_usd = _estimated_cost_usd(
            usage,
            cost_per_1k_tokens_usd=self.cost_per_1k_tokens_usd,
            cost_cny=cost_cny,
            cny_to_usd=self.cny_to_usd,
        )
        model = str(task.get("model") or self.model)
        if self.cost_per_1k_tokens_usd is None:
            cost_tracking = {
                "usage": dict(usage),
                "native_cost": round(cost_cny, 4),
                "native_currency": "CNY",
                "usd_conversion_rate": self.cny_to_usd,
                "estimate": True,
                "usage_missing": not bool(
                    usage.get("completion_tokens") or usage.get("total_tokens")
                ),
                "pricing": {
                    "model": model,
                    "completion_per_million": self.cost_per_million_tokens_cny,
                    "source": self.pricing_source,
                    "as_of": self.pricing_as_of,
                },
            }
        else:
            cost_tracking = {
                "usage": dict(usage),
                "native_cost": cost_usd,
                "native_currency": "USD",
                "estimate": True,
                "usage_missing": not bool(
                    usage.get("completion_tokens") or usage.get("total_tokens")
                ),
                "pricing": {
                    "model": model,
                    "completion_per_1k": self.cost_per_1k_tokens_usd,
                    "source": self.pricing_source,
                    "as_of": self.pricing_as_of,
                },
            }

        return Generation(
            content=str(out_path),
            provider=self.name,
            model=model,
            cost_usd=cost_usd,
            seconds=elapsed,
            meta={
                "task_id": task_id,
                "status": task.get("status"),
                "video_url": video_url,
                "last_frame_url": _last_frame_url(task),
                "usage": usage,
                "cost_tracking": cost_tracking,
                "estimated_cost_cny": round(cost_cny, 4) if cost_cny else 0.0,
                "cost_per_million_tokens_cny": self.cost_per_million_tokens_cny,
                "cny_to_usd": self.cny_to_usd,
                "cost_per_1k_tokens_usd": self.cost_per_1k_tokens_usd,
                "duration_s": task.get("duration", duration),
                "resolution": task.get("resolution", resolution),
                "aspect_ratio": task.get("ratio", ratio),
                "generate_audio": task.get("generate_audio", generate_audio),
                "return_last_frame": body["return_last_frame"],
                "keyframe_path": str(keyframe_path),
                "reference_images": reference_images,
                "last_frame_ref": last_frame_ref,
                "dropped_reference_images": requested_reference_images,
                "dropped_last_frame_ref": requested_last_frame_ref,
            },
        )

    def _poll_task(self, task_id: str, api_key: str) -> dict[str, Any]:
        url = f"{self.base_url}/contents/generations/tasks/{quote(task_id)}"
        last_task: dict[str, Any] | None = None
        for _ in range(self.max_polls):
            task = self._requester(
                "GET",
                url,
                api_key=api_key,
                timeout_s=self.timeout_s,
            )
            last_task = task
            status = str(task.get("status", "")).lower()
            if status == "succeeded":
                return task
            if status in TERMINAL_FAILURE_STATUSES:
                raise RuntimeError(
                    f"{self.provider_label} video task {task_id} {status}: {_error_message(task)}"
                )
            if status not in PENDING_STATUSES:
                raise RuntimeError(
                    f"{self.provider_label} video task {task_id} returned unknown status: {status}"
                )
            if self.poll_interval_s > 0:
                time.sleep(self.poll_interval_s)
        status = (last_task or {}).get("status", "unknown")
        raise TimeoutError(
            f"{self.provider_label} video task {task_id} did not finish; last status: {status}"
        )


def _image_inputs(
    keyframe_path: str,
    reference_images: list[str],
    last_frame_ref: str | None,
) -> list[tuple[str, str]]:
    inputs = [(str(keyframe_path), "first_frame")]
    inputs.extend((str(path), "reference_image") for path in reference_images)
    if last_frame_ref:
        inputs.append((str(last_frame_ref), "reference_image"))
    return inputs


def _seedance_duration(duration_s: Any) -> int:
    # Seedance 2.0 accepts integer durations from 4 to 15 seconds or -1 (model default).
    # ``None`` means the caller wants the model's own default length, so it maps to -1.
    # Keep Studio Agent's shorter fake shot durations valid by clamping them upward for
    # live calls.
    if duration_s is None:
        return -1
    duration = int(round(float(duration_s or 4.0)))
    if duration == -1:
        return duration
    return min(15, max(4, duration))


def _guard_inputs(paths: list[str], max_inputs: int, *, model: str, provider_label: str) -> None:
    if len(paths) > max_inputs:
        raise ValueError(
            f"{model} accepts at most {max_inputs} image inputs, but "
            f"{len(paths)} were passed."
        )
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"{provider_label} video input file not found: {missing[0]}")


def _supports_seed(model: str) -> bool:
    return "seedance-2-0" not in model.lower()


def _supports_native_audio(model: str) -> bool:
    normalized = model.lower().replace("_", "-")
    return "seedance-2-0" in normalized or "seedance-1-5" in normalized


def _data_url(path: str) -> str:
    file_path = Path(path)
    mime_type, _encoding = mimetypes.guess_type(file_path.name)
    if not mime_type or not mime_type.startswith("image/"):
        mime_type = "image/png"
    encoded = base64.b64encode(file_path.read_bytes()).decode("ascii")
    return f"data:{mime_type.lower()};base64,{encoded}"


def _video_url(task: dict[str, Any]) -> str:
    content = task.get("content") if isinstance(task, dict) else None
    url = content.get("video_url") if isinstance(content, dict) else None
    if not url:
        raise RuntimeError("Ark task response did not include content.video_url")
    return str(url)


def _last_frame_url(task: dict[str, Any]) -> str | None:
    content = task.get("content") if isinstance(task, dict) else None
    url = content.get("last_frame_url") if isinstance(content, dict) else None
    return str(url) if url else None


def _estimated_cost_cny(
    usage: dict[str, Any],
    *,
    cost_per_million_tokens_cny: float | None,
) -> float:
    if not cost_per_million_tokens_cny:
        return 0.0
    tokens = usage.get("completion_tokens") or usage.get("total_tokens")
    if tokens is None:
        return 0.0
    return float(tokens) / 1_000_000 * float(cost_per_million_tokens_cny)


def _estimated_cost_usd(
    usage: dict[str, Any],
    *,
    cost_per_1k_tokens_usd: float | None,
    cost_cny: float,
    cny_to_usd: float,
) -> float:
    if cost_per_1k_tokens_usd is None:
        return round(cost_cny * cny_to_usd, 4) if cost_cny else 0.0
    tokens = usage.get("completion_tokens") or usage.get("total_tokens")
    if tokens is None:
        return 0.0
    return round(float(tokens) / 1_000 * float(cost_per_1k_tokens_usd), 4)


# Ark's input content moderation rejects the keyframe (HTTP 400) when its privacy
# filter believes the image may depict a real person — common false positive on a
# photoreal AI keyframe. Map it to actionable guidance instead of a raw JSON dump.
_INPUT_MODERATION_MARKERS = (
    "inputimagesensitivecontentdetected",
    "privacyinformation",
    "may contain real person",
)


def _is_input_moderation_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in _INPUT_MODERATION_MARKERS)


def _friendly_input_moderation_error(exc: Exception, provider_label: str) -> Exception:
    """Return an actionable error for the privacy/moderation rejection, else ``exc``."""
    message = str(exc)
    if not _is_input_moderation_error(exc):
        return exc
    return RuntimeError(
        f"{provider_label} rejected the input keyframe: its privacy filter flagged the "
        "image as possibly containing a real person, so image-to-video was refused. "
        "This is usually a false positive on a photoreal AI keyframe. Regenerate the "
        "keyframe with a more stylized / non-photoreal look (illustrated, painterly, or "
        "anime), or move this shot to a video provider without the privacy filter, then "
        "retry.\nOriginal error: " + message
    )


def _error_message(task: dict[str, Any]) -> str:
    error = task.get("error")
    if isinstance(error, dict):
        message = error.get("message") or error.get("code")
        if message:
            return str(message)
    return json.dumps(task, ensure_ascii=False)


def _is_transient_urlerror(exc: URLError) -> bool:
    """A connection-phase failure worth retrying: a DNS lookup miss or a dropped/refused
    connection. A connection *timeout* is handled separately (kept terminal + actionable)."""
    reason = getattr(exc, "reason", None)
    return isinstance(reason, (socket.gaierror, ConnectionError))


def _request_json(
    method: str,
    url: str,
    *,
    api_key: str,
    json_body: dict[str, Any] | None = None,
    timeout_s: float = 60.0,
    max_retries: int = MAX_NETWORK_RETRIES,
    backoff_s: float = RETRY_BACKOFF_S,
) -> dict[str, Any]:
    data = None
    if json_body is not None:
        data = json.dumps(json_body).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    request = Request(url, data=data, headers=headers, method=method)
    attempt = 0
    while True:
        try:
            with urlopen(request, timeout=timeout_s) as response:
                payload = response.read().decode("utf-8")
            break
        except HTTPError as exc:
            # 5xx is a transient server-side failure; 4xx (auth, bad request) is terminal.
            if exc.code >= 500 and attempt < max_retries:
                attempt += 1
                time.sleep(backoff_s * (2 ** (attempt - 1)))
                continue
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ark API HTTP {exc.code}: {detail}") from exc
        except TimeoutError as exc:
            # A read-phase socket timeout raises a bare TimeoutError ("The read
            # operation timed out") that is NOT a URLError subclass, so it must be
            # handled explicitly. Synchronous Ark image/video renders can legitimately
            # run long; surface an actionable remedy instead of the bare message.
            raise RuntimeError(
                f"Ark API request timed out after {timeout_s:g}s waiting for the response. "
                "Heavy renders (2K turnaround/expression sheets) can exceed the default; "
                "raise image_timeout_s / video_timeout_s in the model config and re-run "
                "(the stage backfills only the missing artifact)."
            ) from exc
        except URLError as exc:
            # URLError wraps connection-phase socket timeouts as exc.reason.
            if isinstance(getattr(exc, "reason", None), TimeoutError):
                raise RuntimeError(
                    f"Ark API connection timed out after {timeout_s:g}s. "
                    "Check connectivity or raise image_timeout_s / video_timeout_s and re-run."
                ) from exc
            # A transient DNS/connection blip must not abort a long, already-charged paid
            # run — ride through a few with backoff before surfacing the failure.
            if _is_transient_urlerror(exc) and attempt < max_retries:
                attempt += 1
                time.sleep(backoff_s * (2 ** (attempt - 1)))
                continue
            raise RuntimeError(f"Ark API request failed: {exc.reason}") from exc
    if not payload:
        return {}
    return json.loads(payload)


def _download_url(url: str, out_path: str) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response, path.open("wb") as out:
        shutil.copyfileobj(response, out)
