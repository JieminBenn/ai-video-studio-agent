"""OpenAI GPT Image provider.

Uses the Images API directly with ``gpt-image-2`` by default. Text-only keyframes go
through ``/images/generations``; calls with reference images use ``/images/edits`` so
uploaded character/location/style references can condition outputs without changing
stage logic.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .base import Generation, ImageGen

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-image-2"
DEFAULT_SIZE = "1536x1024"
DEFAULT_QUALITY = "medium"
MAX_REFERENCE_IMAGES = 10
MAX_PROMPT_LENGTH = 32000
PROMPT_LENGTH_UNIT = "characters"


def _default_max_prompt_length(model: str) -> int | None:
    normalized = str(model or "").strip().lower()
    if normalized == "dall-e-2":
        return 1000
    if normalized == "dall-e-3":
        return 4000
    if normalized.startswith("gpt-image-"):
        return MAX_PROMPT_LENGTH
    return None


JsonRequester = Callable[..., dict[str, Any]]
MultipartRequester = Callable[..., dict[str, Any]]
Downloader = Callable[[str, str], None]


class OpenAIImageGen(ImageGen):
    name = "openai"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        base_url: str = DEFAULT_BASE_URL,
        size: str = DEFAULT_SIZE,
        quality: str | None = DEFAULT_QUALITY,
        api_key_env: str = "OPENAI_API_KEY",
        cost_per_image: float = 0.08,
        max_reference_images: int = MAX_REFERENCE_IMAGES,
        timeout_s: float = 120.0,
        json_requester: JsonRequester | None = None,
        multipart_requester: MultipartRequester | None = None,
        downloader: Downloader | None = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.size = size
        self.quality = quality
        self.api_key_env = api_key_env
        self.cost_per_image = cost_per_image
        self.max_reference_images = max_reference_images
        self.max_prompt_length = _default_max_prompt_length(model)
        self.prompt_length_unit = PROMPT_LENGTH_UNIT
        self.timeout_s = timeout_s
        self._json_requester = json_requester or _request_json
        self._multipart_requester = multipart_requester or _request_multipart
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
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"{self.api_key_env} not set - add it to .env (see `cli doctor`).")

        size = kwargs.get("size") or kwargs.get("image_size") or self.size
        quality = kwargs.get("quality") or kwargs.get("image_quality") or self.quality
        body_fields = {
            "model": self.model,
            "prompt": prompt.strip(),
            "size": str(size),
        }
        if quality:
            body_fields["quality"] = str(quality)

        start = time.time()
        if refs:
            files = [
                (
                    "image[]",
                    Path(ref).name,
                    Path(ref).read_bytes(),
                    _mime_type(ref),
                )
                for ref in refs
            ]
            response = self._multipart_requester(
                f"{self.base_url}/images/edits",
                api_key=api_key,
                fields=body_fields,
                files=files,
                timeout_s=self.timeout_s,
            )
            endpoint = "edits"
        else:
            response = self._json_requester(
                f"{self.base_url}/images/generations",
                api_key=api_key,
                json_body=body_fields,
                timeout_s=self.timeout_s,
            )
            endpoint = "generations"

        image_meta = _write_first_image(response, out_path, downloader=self._downloader)
        elapsed = round(time.time() - start, 2)
        return Generation(
            content=str(out_path),
            provider=self.name,
            model=self.model,
            cost_usd=float(kwargs.get("cost_per_image", self.cost_per_image)),
            seconds=elapsed,
            meta={
                "endpoint": endpoint,
                "seed": kwargs.get("seed"),
                "size": size,
                "quality": quality,
                "reference_images": refs,
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
        raise FileNotFoundError(f"OpenAI image reference file not found: {missing[0]}")


def _write_first_image(
    response: dict[str, Any],
    out_path: str,
    *,
    downloader: Downloader,
) -> dict[str, Any]:
    images = response.get("data") if isinstance(response, dict) else None
    if not isinstance(images, list) or not images:
        raise RuntimeError("OpenAI Images API returned no image data.")
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
    raise RuntimeError("OpenAI Images API response had no b64_json or url field.")


def _request_json(
    url: str,
    *,
    api_key: str,
    json_body: dict[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    data = json.dumps(json_body).encode("utf-8")
    request = Request(
        url,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    return _read_response(request, timeout_s)


def _request_multipart(
    url: str,
    *,
    api_key: str,
    fields: dict[str, str],
    files: list[tuple[str, str, bytes, str]],
    timeout_s: float,
) -> dict[str, Any]:
    boundary = f"----studio-agent-{uuid.uuid4().hex}"
    data = _encode_multipart(boundary, fields, files)
    request = Request(
        url,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(data)),
        },
    )
    return _read_response(request, timeout_s)


def _read_response(request: Request, timeout_s: float) -> dict[str, Any]:
    try:
        with urlopen(request, timeout=timeout_s) as response:
            payload = response.read().decode("utf-8")
    except HTTPError as exc:  # pragma: no cover - live API failures only
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI Images API HTTP {exc.code}: {detail}") from exc
    except URLError as exc:  # pragma: no cover - live network failures only
        raise RuntimeError(f"OpenAI Images API request failed: {exc.reason}") from exc
    return json.loads(payload) if payload else {}


def _encode_multipart(
    boundary: str,
    fields: dict[str, str],
    files: list[tuple[str, str, bytes, str]],
) -> bytes:
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )
    for field_name, filename, data, mime_type in files:
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                (
                    f'Content-Disposition: form-data; name="{field_name}"; '
                    f'filename="{filename}"\r\n'
                ).encode("utf-8"),
                f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8"),
                data,
                b"\r\n",
            ]
        )
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks)


def _download_url(url: str, out_path: str) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response, path.open("wb") as out:
        shutil.copyfileobj(response, out)


def _mime_type(path: str) -> str:
    mime_type, _encoding = mimetypes.guess_type(Path(path).name)
    if not mime_type or not mime_type.startswith("image/"):
        return "image/png"
    return mime_type
