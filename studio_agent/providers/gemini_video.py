"""Native Gemini API Veo 3.1 video generation.

The google-genai dependency and key are loaded only for a live generation call. Tests
inject the operation responder, image loader, and downloader.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from .base import Generation, VideoCapabilities, VideoGen

DEFAULT_MODEL = "veo-3.1-generate-preview"


class GeminiVeoVideoGen(VideoGen):
    name = "gemini-veo"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        api_key_env: str = "GEMINI_API_KEY",
        resolution: str = "720p",
        aspect_ratio: str = "16:9",
        generate_audio: bool = True,
        cost_per_second: float = 0.0,
        poll_interval_s: float = 10.0,
        max_polls: int = 180,
        supports_reference_images: bool = True,
        supports_last_frame: bool = True,
        max_image_inputs: int = 4,
        responder=None,
        poller=None,
        image_loader=None,
        downloader=None,
    ):
        self.model = model
        self.api_key_env = api_key_env
        self.resolution = resolution
        self.aspect_ratio = aspect_ratio
        self.generate_audio = generate_audio
        self.cost_per_second = cost_per_second
        self.poll_interval_s = poll_interval_s
        self.max_polls = max_polls
        self.supports_reference_images = supports_reference_images
        self.supports_last_frame = supports_last_frame
        self.max_image_inputs = max_image_inputs
        self._responder = responder
        self._poller = poller
        self._image_loader = image_loader
        self._downloader = downloader
        self._client = None

    @property
    def capabilities(self) -> VideoCapabilities:
        return VideoCapabilities(
            supports_reference_images=self.supports_reference_images,
            supports_last_frame=self.supports_last_frame,
            supports_native_audio=self.generate_audio,
            max_image_inputs=self.max_image_inputs,
            supports_storyboard_grid=getattr(self, "_supports_storyboard_grid", False),
            # Veo 3.x accepts durationSeconds 4/6/8 (forced to 8 with reference images);
            # Veo 2 accepts 5-8. ``None`` omits the field so the service default governs.
            min_duration_s=4,
            max_duration_s=8,
        )

    def _client_lazy(self):
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("google-genai not installed — run `pip install -e \".[real]\"`.") from exc
            key = os.environ.get(self.api_key_env)
            if not key:
                raise RuntimeError(f"{self.api_key_env} not set - add it to .env (see `cli doctor`).")
            self._client = genai.Client(api_key=key)
        return self._client

    def generate(self, prompt: str, *, out_path: str, **kwargs: Any) -> Generation:
        keyframe_path = kwargs.get("keyframe_path")
        if not keyframe_path:
            raise ValueError("Gemini Veo image-to-video requires keyframe_path.")
        requested_refs = list(kwargs.get("reference_images") or [])
        requested_last_frame = kwargs.get("last_frame_ref")
        refs = requested_refs if self.supports_reference_images else []
        last_frame = requested_last_frame if self.supports_last_frame else None
        inputs = [str(keyframe_path), *refs, *([str(last_frame)] if last_frame else [])]
        if len(inputs) > self.capabilities.max_image_inputs:
            raise ValueError(
                f"{self.model} accepts at most {self.capabilities.max_image_inputs} image inputs, "
                f"but {len(inputs)} were passed."
            )
        missing = [path for path in inputs if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"Gemini Veo input file not found: {missing[0]}")
        if self._responder is None and not os.environ.get(self.api_key_env):
            raise RuntimeError(f"{self.api_key_env} not set - add it to .env (see `cli doctor`).")

        image_loader = self._image_loader or self._load_image
        first_image = image_loader(str(keyframe_path))
        reference_images = [image_loader(path) for path in refs]
        last_image = image_loader(str(last_frame)) if last_frame else None
        raw_duration = kwargs.get("duration_s")
        duration = int(round(float(raw_duration))) if raw_duration else None
        ratio = kwargs.get("aspect_ratio") or self.aspect_ratio
        resolution = kwargs.get("resolution") or self.resolution
        generate_audio = bool(kwargs.get("generate_audio", self.generate_audio))

        start = time.time()
        operation = self._start_operation(
            prompt=prompt.strip(),
            image=first_image,
            reference_images=reference_images,
            last_frame=last_image,
            duration_s=duration,
            aspect_ratio=ratio,
            resolution=resolution,
            generate_audio=generate_audio,
        )
        operation = self._wait(operation)
        generated = self._generated_video(operation)
        self._save_video(generated, out_path)
        operation_name = getattr(operation, "name", None)
        video = getattr(generated, "video", generated)
        # When the duration was omitted the service picks the length; bill at the model
        # maximum so the estimate never undercounts.
        billed_seconds = duration if duration is not None else self.capabilities.max_duration_s
        return Generation(
            content=str(out_path),
            provider=self.name,
            model=self.model,
            cost_usd=round(billed_seconds * self.cost_per_second, 4),
            seconds=round(time.time() - start, 2),
            meta={
                "operation_name": operation_name,
                "video_url": getattr(video, "uri", None),
                "duration_s": duration,
                "resolution": resolution,
                "aspect_ratio": ratio,
                "generate_audio": generate_audio,
                "keyframe_path": str(keyframe_path),
                "reference_images": refs,
                "last_frame_ref": str(last_frame) if last_frame else None,
            },
        )

    def _start_operation(self, **kwargs):
        if self._responder is not None:
            return self._responder(model=self.model, **kwargs)

        from google.genai import types

        references = [
            types.VideoGenerationReferenceImage(image=image, reference_type="asset")
            for image in kwargs.pop("reference_images")
        ]
        image = kwargs.pop("image")
        prompt = kwargs.pop("prompt")
        last_frame = kwargs.pop("last_frame")
        config_kwargs: dict[str, Any] = {
            "aspect_ratio": kwargs.pop("aspect_ratio"),
            "resolution": kwargs.pop("resolution"),
            "generate_audio": kwargs.pop("generate_audio"),
            "reference_images": references or None,
            "last_frame": last_frame,
        }
        duration_s = kwargs.pop("duration_s")
        if duration_s is not None:
            # Omitted entirely when the caller wants the model-default clip length.
            config_kwargs["duration_seconds"] = duration_s
        config = types.GenerateVideosConfig(**config_kwargs)
        return self._client_lazy().models.generate_videos(
            model=self.model,
            prompt=prompt,
            image=image,
            config=config,
        )

    def _wait(self, operation):
        for _ in range(self.max_polls):
            if getattr(operation, "done", False):
                return operation
            if self.poll_interval_s > 0:
                time.sleep(self.poll_interval_s)
            if self._poller is not None:
                operation = self._poller(operation)
            else:
                operation = self._client_lazy().operations.get(operation)
        raise TimeoutError(f"Gemini Veo operation did not finish after {self.max_polls} polls.")

    @staticmethod
    def _generated_video(operation):
        response = getattr(operation, "response", None)
        videos = getattr(response, "generated_videos", None) or []
        if not videos:
            error = getattr(operation, "error", None)
            raise RuntimeError(f"Gemini Veo operation returned no generated video: {error or 'unknown error'}")
        return videos[0]

    def _save_video(self, generated, out_path: str) -> None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        video = getattr(generated, "video", generated)
        if self._downloader is not None:
            self._downloader(video, str(path))
            return
        client = self._client_lazy()
        client.files.download(file=video)
        video.save(str(path))

    @staticmethod
    def _load_image(path: str):
        from google.genai import types

        return types.Image.from_file(location=path)
