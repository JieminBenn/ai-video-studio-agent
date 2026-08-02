"""Volcengine Ark / BytePlus Seedream image generation provider.

Reference docs verified 2026-06-19:
- https://docs.byteplus.com/en/docs/ModelArk/1824121
- https://docs.byteplus.com/en/docs/ModelArk/1541523
"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any, Callable

from .base import Generation, ImageGen
from .volcengine import (
    BYTEPLUS_BASE_URL,
    DEFAULT_BASE_URL,
    DEFAULT_CNY_TO_USD,
    _data_url,
    _download_url,
    _estimated_cost_usd,
    _request_json,
)

DEFAULT_MODEL = "doubao-seedream-4-0-250828"
BYTEPLUS_DEFAULT_MODEL = "seedream-4-0-250828"
DEFAULT_SIZE = "2K"
DEFAULT_ASPECT_RATIO = "16:9"
DEFAULT_RESPONSE_FORMAT = "url"
DEFAULT_COST_PER_IMAGE_CNY = 0.2
BYTEPLUS_DEFAULT_COST_PER_IMAGE_USD = 0.03
MAX_REFERENCE_IMAGES = 6

MODEL_ALIASES = {
    "doubao-seedream-4.0": "doubao-seedream-4-0-250828",
    "doubao-seedream-4-0": "doubao-seedream-4-0-250828",
    "doubao-seedream-4.5": "doubao-seedream-4-5-251128",
    "doubao-seedream-4-5": "doubao-seedream-4-5-251128",
    "doubao-seedream-5.0": "doubao-seedream-5-0-260128",
    "doubao-seedream-5-0": "doubao-seedream-5-0-260128",
    "doubao-seedream-5.0-lite": "doubao-seedream-5-0-lite-260128",
    "doubao-seedream-5-0-lite": "doubao-seedream-5-0-lite-260128",
    "seedream-4.0": "seedream-4-0-250828",
    "seedream-4-0": "seedream-4-0-250828",
    "seedream-4.5": "seedream-4-5-251128",
    "seedream-4-5": "seedream-4-5-251128",
    "seedream-5.0": "seedream-5-0-260128",
    "seedream-5-0": "seedream-5-0-260128",
    "seedream-5.0-lite": "seedream-5-0-lite-260128",
    "seedream-5-0-lite": "seedream-5-0-lite-260128",
}

Requester = Callable[..., dict[str, Any]]
Downloader = Callable[[str, str], None]


class ArkSeedreamImageGen(ImageGen):
    name = "doubao"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        provider_name: str = "doubao",
        provider_label: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        api_key_env: str = "ARK_API_KEY",
        size: str = DEFAULT_SIZE,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        output_format: str | None = None,
        max_reference_images: int = MAX_REFERENCE_IMAGES,
        poll_interval_s: float = 2.0,
        max_polls: int = 120,
        timeout_s: float = 120.0,
        cost_per_image_cny: float | None = DEFAULT_COST_PER_IMAGE_CNY,
        cost_per_image_usd: float | None = None,
        cny_to_usd: float = DEFAULT_CNY_TO_USD,
        requester: Requester | None = None,
        downloader: Downloader | None = None,
    ):
        self.name = provider_name
        self.provider_label = provider_label or provider_name
        self.requested_model = model
        self.model = _normalize_model(model)
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.size = size
        self.aspect_ratio = aspect_ratio
        self.output_format = output_format
        self.max_reference_images = max_reference_images
        self.poll_interval_s = poll_interval_s
        self.max_polls = max_polls
        self.timeout_s = timeout_s
        self.cost_per_image_cny = cost_per_image_cny
        self.cost_per_image_usd = cost_per_image_usd
        self.cny_to_usd = cny_to_usd
        self._requester = requester or _request_images_json
        self._downloader = downloader or _download_url

    def generate(
        self,
        prompt: str,
        *,
        out_path: str,
        reference_images: list[str] | None = None,
        **kwargs: Any,
    ) -> Generation:
        refs = list(reference_images or [])
        _guard_reference_images(refs, self.max_reference_images, self.model)

        import os

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"{self.api_key_env} not set - add it to .env (see `cli doctor`).")

        size = str(kwargs.get("size") or kwargs.get("image_size") or self.size)
        # `output_format` is opt-in: several Ark/Seedream models (e.g.
        # doubao-seedream-4-0) reject it with HTTP 400, so only send it when a
        # provider/config explicitly asks for one.
        output_format = kwargs.get("output_format") or self.output_format
        response_format = str(kwargs.get("response_format") or DEFAULT_RESPONSE_FORMAT)
        reference_payload = [_data_url(ref) for ref in refs]
        body = {
            "model": self.model,
            "prompt": prompt.strip(),
            "size": size,
            "stream": False,
            "response_format": response_format,
            "watermark": False,
        }
        if output_format:
            body["output_format"] = str(output_format)
        if reference_payload:
            body["image"] = (
                reference_payload[0] if len(reference_payload) == 1 else reference_payload
            )
        if kwargs.get("sequential_image_generation") is not None:
            body["sequential_image_generation"] = str(kwargs["sequential_image_generation"])
        if kwargs.get("sequential_image_generation_options") is not None:
            body["sequential_image_generation_options"] = kwargs[
                "sequential_image_generation_options"
            ]
        if kwargs.get("optimize_prompt_options") is not None:
            body["optimize_prompt_options"] = kwargs["optimize_prompt_options"]
        ratio = kwargs.get("aspect_ratio") or kwargs.get("ratio")
        if ratio:
            body["ratio"] = str(ratio)
        if kwargs.get("seed") is not None:
            body["seed"] = int(kwargs["seed"])

        start = time.time()
        response = self._requester(
            f"{self.base_url}/images/generations",
            api_key=api_key,
            json_body=body,
            timeout_s=self.timeout_s,
        )

        image_meta = _write_first_image(response, out_path, downloader=self._downloader)
        elapsed = round(time.time() - start, 2)

        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        cost_usd = _cost_usd(
            usage,
            cost_per_image_usd=self.cost_per_image_usd,
            cost_per_image_cny=self.cost_per_image_cny,
            cny_to_usd=self.cny_to_usd,
        )
        return Generation(
            content=str(out_path),
            provider=self.name,
            model=str(response.get("model") or self.model),
            cost_usd=cost_usd,
            seconds=elapsed,
            meta={
                "endpoint": "images/generations",
                "requested_model": self.requested_model,
                "usage": usage,
                "size": size,
                "aspect_ratio": kwargs.get("aspect_ratio") or kwargs.get("ratio") or self.aspect_ratio,
                "output_format": output_format,
                "response_format": response_format,
                "reference_images": refs,
                "cost_per_image_cny": self.cost_per_image_cny,
                "cost_per_image_usd": self.cost_per_image_usd,
                **image_meta,
            },
        )


def _guard_reference_images(refs: list[str], max_reference_images: int, model: str) -> None:
    if len(refs) > max_reference_images:
        raise ValueError(
            f"{model} accepts at most {max_reference_images} reference images, "
            f"but {len(refs)} were passed."
        )
    missing = [ref for ref in refs if not Path(ref).is_file()]
    if missing:
        raise FileNotFoundError(f"Seedream image reference file not found: {missing[0]}")


def _normalize_model(model: str) -> str:
    value = str(model or "").strip()
    return MODEL_ALIASES.get(value.lower(), value)


def _request_images_json(
    url: str,
    *,
    api_key: str,
    json_body: dict[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    return _request_json(
        "POST",
        url,
        api_key=api_key,
        json_body=json_body,
        timeout_s=timeout_s,
    )


def _write_first_image(
    response: dict[str, Any],
    out_path: str,
    *,
    downloader: Downloader,
) -> dict[str, Any]:
    images = response.get("data") if isinstance(response, dict) else None
    if not isinstance(images, list) or not images:
        raise RuntimeError("Seedream Images API returned no image data.")
    first = images[0]
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(first, dict) and first.get("b64_json"):
        path.write_bytes(base64.b64decode(str(first["b64_json"])))
        return {"image_response": "b64_json"}
    if isinstance(first, dict) and first.get("url"):
        url = str(first["url"])
        downloader(url, str(path))
        return {"image_response": "url", "image_url": url}
    raise RuntimeError("Seedream Images API response had no b64_json or url field.")


def _cost_usd(
    usage: dict[str, Any],
    *,
    cost_per_image_usd: float | None,
    cost_per_image_cny: float | None,
    cny_to_usd: float,
) -> float:
    if cost_per_image_usd is not None:
        return round(float(cost_per_image_usd), 4)
    if usage:
        usage_cost = _estimated_cost_usd(
            usage,
            cost_per_1k_tokens_usd=None,
            cost_cny=0.0,
            cny_to_usd=cny_to_usd,
        )
        if usage_cost:
            return usage_cost
    if cost_per_image_cny is None:
        return 0.0
    return round(float(cost_per_image_cny) * float(cny_to_usd), 4)
