"""Direct xAI Grok Imagine image-to-video provider."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .base import Generation, VideoCapabilities, VideoGen

DEFAULT_BASE_URL = "https://api.x.ai/v1"
DEFAULT_MODEL = "grok-imagine-video"
PENDING_STATUSES = {"queued", "pending", "running", "processing"}
FAILED_STATUSES = {"failed", "expired", "cancelled"}

Requester = Callable[..., dict[str, Any]]
Downloader = Callable[[str, str], None]


class XAIVideoGen(VideoGen):
    name = "xai"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        base_url: str = DEFAULT_BASE_URL,
        api_key_env: str = "XAI_API_KEY",
        resolution: str = "720p",
        aspect_ratio: str = "16:9",
        cost_per_second: float = 0.07,
        cost_per_input_image: float = 0.002,
        poll_interval_s: float = 5.0,
        max_polls: int = 180,
        timeout_s: float = 60.0,
        requester: Requester | None = None,
        downloader: Downloader | None = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.resolution = resolution
        self.aspect_ratio = aspect_ratio
        self.cost_per_second = cost_per_second
        self.cost_per_input_image = cost_per_input_image
        self.poll_interval_s = poll_interval_s
        self.max_polls = max_polls
        self.timeout_s = timeout_s
        self._requester = requester or _request_json
        self._downloader = downloader or _download_url

    @property
    def capabilities(self) -> VideoCapabilities:
        return VideoCapabilities(
            supports_reference_images=False,
            supports_last_frame=False,
            max_image_inputs=1,
            supports_storyboard_grid=getattr(self, "_supports_storyboard_grid", False),
            min_duration_s=1,
            max_duration_s=15,
            # Omitting the duration is unverified against the live API, so ``None`` from
            # the caller resolves to this default instead of being dropped from the body.
            default_duration_s=5,
        )

    def generate(self, prompt: str, *, out_path: str, **kwargs: Any) -> Generation:
        keyframe_path = kwargs.get("keyframe_path")
        if not keyframe_path:
            raise ValueError("xAI Grok Imagine image-to-video requires keyframe_path.")
        if not Path(keyframe_path).is_file():
            raise FileNotFoundError(f"xAI video keyframe not found: {keyframe_path}")

        api_key = os.environ.get(self.api_key_env)
        if not api_key and self._requester is _request_json:
            raise RuntimeError(f"{self.api_key_env} not set - add it to .env (see `cli doctor`).")

        duration = min(15, max(1, int(round(float(kwargs.get("duration_s") or 5)))))
        resolution = kwargs.get("resolution") or self.resolution
        ratio = kwargs.get("aspect_ratio") or self.aspect_ratio
        body = {
            "model": self.model,
            "prompt": prompt.strip(),
            "image": {"url": _data_url(str(keyframe_path))},
            "duration": duration,
            "aspect_ratio": ratio,
            "resolution": resolution,
        }

        start = time.time()
        create = self._requester(
            "POST",
            f"{self.base_url}/videos/generations",
            api_key=api_key or "",
            json_body=body,
            timeout_s=self.timeout_s,
        )
        request_id = create.get("request_id") if isinstance(create, dict) else None
        if not request_id:
            raise RuntimeError("xAI video create response did not include request_id.")
        result = self._poll(str(request_id), api_key or "")
        video_url = _video_url(result)
        self._downloader(video_url, out_path)
        return Generation(
            content=str(out_path),
            provider=self.name,
            model=self.model,
            cost_usd=round(duration * self.cost_per_second + self.cost_per_input_image, 4),
            seconds=round(time.time() - start, 2),
            meta={
                "request_id": request_id,
                "video_url": video_url,
                "status": result.get("status"),
                "duration_s": duration,
                "resolution": resolution,
                "aspect_ratio": ratio,
                "keyframe_path": str(keyframe_path),
                "reference_images": [],
                "last_frame_ref": None,
                "cost_per_input_image": self.cost_per_input_image,
            },
        )

    def _poll(self, request_id: str, api_key: str) -> dict[str, Any]:
        for _ in range(self.max_polls):
            result = self._requester(
                "GET",
                f"{self.base_url}/videos/{quote(request_id)}",
                api_key=api_key,
                timeout_s=self.timeout_s,
            )
            status = str(result.get("status") or "").lower()
            if status == "done":
                return result
            if status in FAILED_STATUSES:
                raise RuntimeError(f"xAI video request {request_id} {status}: {result.get('error') or result}")
            if status not in PENDING_STATUSES:
                raise RuntimeError(f"xAI video request {request_id} returned unknown status: {status}")
            if self.poll_interval_s > 0:
                time.sleep(self.poll_interval_s)
        raise TimeoutError(f"xAI video request {request_id} did not finish after {self.max_polls} polls.")


def _data_url(path: str) -> str:
    file_path = Path(path)
    mime, _encoding = mimetypes.guess_type(file_path.name)
    if not mime or not mime.startswith("image/"):
        mime = "image/png"
    return f"data:{mime};base64,{base64.b64encode(file_path.read_bytes()).decode('ascii')}"


def _video_url(result: dict[str, Any]) -> str:
    video = result.get("video") if isinstance(result, dict) else None
    url = video.get("url") if isinstance(video, dict) else None
    if not url:
        raise RuntimeError("xAI video response did not include video.url.")
    return str(url)


def _request_json(
    method: str,
    url: str,
    *,
    api_key: str,
    json_body: dict[str, Any] | None = None,
    timeout_s: float,
) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(json_body).encode("utf-8") if json_body is not None else None,
        method=method,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"xAI video API HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"xAI video API request failed: {exc.reason}") from exc


def _download_url(url: str, out_path: str) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # xAI serves the generated video from its imgen.x.ai CDN, which rejects urllib's
    # default "Python-urllib/x.y" User-Agent with HTTP 403 Forbidden. Send a browser-like
    # User-Agent so the download succeeds (the URL itself is unauthenticated).
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 (studio-agent)"})
    with urlopen(request) as response, path.open("wb") as output:
        shutil.copyfileobj(response, output)
