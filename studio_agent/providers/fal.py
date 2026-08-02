"""fal.ai reference-to-video provider.

This provider targets Seedance 2 reference-to-video on fal. It is lazy like the other
real providers: importing or constructing it needs no SDK, network, or key. A live
``generate()`` requires ``fal-client`` and ``FAL_KEY``.

Reference docs verified 2026-06-16:
- https://fal.ai/models/bytedance/seedance-2.0/reference-to-video
- https://fal.ai/docs/documentation/model-apis/fal-cdn
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Callable
from urllib.request import urlopen

from .base import Generation, VideoCapabilities, VideoGen

DEFAULT_MODEL = "bytedance/seedance-2.0/fast/reference-to-video"
DEFAULT_RESOLUTION = "720p"
DEFAULT_ASPECT_RATIO = "16:9"
DEFAULT_COST_PER_SECOND = 0.2419
MAX_IMAGE_INPUTS = 9


class FalReferenceVideoGen(VideoGen):
    name = "fal"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        resolution: str = DEFAULT_RESOLUTION,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        generate_audio: bool = False,
        cost_per_second: float = DEFAULT_COST_PER_SECOND,
        api_key_env: str = "FAL_KEY",
        max_image_inputs: int = MAX_IMAGE_INPUTS,
        downloader: Callable[[str, str], None] | None = None,
    ):
        self.model = model
        self.resolution = resolution
        self.aspect_ratio = aspect_ratio
        self.generate_audio = generate_audio
        self.cost_per_second = cost_per_second
        self.api_key_env = api_key_env
        self.max_image_inputs = max_image_inputs
        self._downloader = downloader or _download_url
        self._client = None

    @property
    def capabilities(self) -> VideoCapabilities:
        return VideoCapabilities(
            supports_native_audio=self.generate_audio,
            max_image_inputs=self.max_image_inputs,
            supports_storyboard_grid=getattr(self, "_supports_storyboard_grid", False),
            min_duration_s=4,
            max_duration_s=15,
        )

    def _client_lazy(self):
        if self._client is None:
            try:
                import fal_client
            except ImportError as exc:  # pragma: no cover - optional extra
                raise RuntimeError(
                    "fal-client not installed — run `pip install -e \".[real]\"`."
                ) from exc
            if not os.environ.get(self.api_key_env):
                raise RuntimeError(
                    f"{self.api_key_env} not set — add it to .env (see `cli doctor`)."
                )
            self._client = fal_client
        return self._client

    def generate(self, prompt: str, *, out_path: str, **kwargs) -> Generation:
        keyframe_path = kwargs.get("keyframe_path")
        if not keyframe_path:
            raise ValueError("fal reference-to-video requires keyframe_path.")

        target_state_reference_images = list(kwargs.get("target_state_reference_images") or [])
        reference_images = _dedupe_paths([
            *list(kwargs.get("reference_images") or []),
            *target_state_reference_images,
        ])
        last_frame_ref = kwargs.get("last_frame_ref")
        image_inputs = [str(keyframe_path), *reference_images]
        if last_frame_ref:
            image_inputs.append(str(last_frame_ref))
        _guard_inputs(image_inputs, self.max_image_inputs)

        duration = _fal_duration(kwargs.get("duration_s", 4.0))
        aspect_ratio = kwargs.get("aspect_ratio") or self.aspect_ratio
        seed = kwargs.get("seed")
        generate_audio = bool(kwargs.get("generate_audio", self.generate_audio))

        client = self._client_lazy()
        image_urls = [client.upload_file(path) for path in image_inputs]
        fal_prompt = _fal_prompt(
            prompt,
            has_refs=bool(reference_images),
            has_last_frame=bool(last_frame_ref),
            target_labels=_target_labels(reference_images, target_state_reference_images),
        )
        arguments = {
            "prompt": fal_prompt,
            "image_urls": image_urls,
            "resolution": self.resolution,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
            "generate_audio": generate_audio,
        }
        if seed is not None:
            arguments["seed"] = int(seed)

        start = time.time()
        result = client.subscribe(self.model, arguments=arguments)
        elapsed = round(time.time() - start, 2)
        video_url = _video_url(result)
        self._downloader(video_url, out_path)

        billed_seconds = _billed_seconds(duration, result)
        return Generation(
            content=str(out_path),
            provider=self.name,
            model=self.model,
            cost_usd=round(billed_seconds * self.cost_per_second, 4),
            seconds=elapsed,
            meta={
                "seed": result.get("seed", seed),
                "duration_s": billed_seconds,
                "resolution": self.resolution,
                "aspect_ratio": aspect_ratio,
                "video_url": video_url,
                "image_urls": image_urls,
                "keyframe_path": str(keyframe_path),
                "reference_images": reference_images,
                "target_state_reference_images": target_state_reference_images,
                "last_frame_ref": last_frame_ref,
                "generate_audio": generate_audio,
            },
        )


def _fal_duration(duration_s) -> str:
    # Seedance accepts integer durations from 4 to 15 seconds or "auto" (model default).
    # ``None`` means the caller wants the model's own default length. Keep Studio
    # Agent's shorter fake shot durations valid by clamping them upward for live calls.
    if duration_s is None:
        return "auto"
    duration = int(round(float(duration_s or 4.0)))
    return str(min(15, max(4, duration)))


def _billed_seconds(duration: str, result: dict) -> float:
    # Duration drives the cost estimate. On "auto" the request omitted a length, so read
    # the seconds the service reports back when present; otherwise estimate with the
    # Seedance range midpoint rather than crashing on the non-numeric sentinel.
    try:
        return float(duration)
    except (TypeError, ValueError):
        pass
    reported = result.get("duration") if isinstance(result, dict) else None
    try:
        if reported is not None:
            return float(reported)
    except (TypeError, ValueError):
        pass
    return (4 + 15) / 2


def _guard_inputs(paths: list[str], max_inputs: int) -> None:
    if len(paths) > max_inputs:
        raise ValueError(
            f"{DEFAULT_MODEL} accepts at most {max_inputs} image inputs, but "
            f"{len(paths)} were passed."
        )
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"fal video input file not found: {missing[0]}")


def _fal_prompt(
    prompt: str,
    *,
    has_refs: bool,
    has_last_frame: bool,
    target_labels: list[str] | None = None,
) -> str:
    target_labels = list(target_labels or [])
    parts = [
        prompt.strip(),
        "Use @Image1 as the starting keyframe and preserve its composition.",
    ]
    if target_labels:
        labels = ", ".join(target_labels)
        parts.append(
            f"Use {labels} as the approved target-state reference image(s); the final "
            "transformed form must match this target-state design exactly."
        )
        parts.append(
            "Use any other supplied reference images to preserve character, location, "
            "and style continuity."
        )
    elif has_refs:
        parts.append("Use @Image2 and any remaining character reference images to preserve identity.")
    if has_last_frame:
        parts.append("Use the final image input as previous-shot continuity guidance.")
    return "\n\n".join(part for part in parts if part)


def _target_labels(reference_images: list[str], target_images: list[str]) -> list[str]:
    targets = {str(Path(path)) for path in target_images}
    return [
        f"@Image{index + 2}"
        for index, path in enumerate(reference_images)
        if str(Path(path)) in targets
    ]


def _dedupe_paths(paths: list[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for path in paths:
        key = str(Path(path))
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _video_url(result: dict) -> str:
    video = result.get("video") if isinstance(result, dict) else None
    url = video.get("url") if isinstance(video, dict) else None
    if not url:
        raise RuntimeError("fal response did not include video.url")
    return url


def _download_url(url: str, out_path: str) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response, path.open("wb") as out:
        shutil.copyfileobj(response, out)
