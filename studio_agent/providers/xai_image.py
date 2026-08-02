"""xAI Grok Imagine image generation and reference-conditioned editing.

The provider uses xAI's OpenAI-shaped Images endpoints but keeps its JSON reference
format isolated here. Construction is offline and keyless; only ``generate`` needs
``XAI_API_KEY`` when no test requester is injected.
"""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import shutil
import socket
import time
from pathlib import Path
from typing import Any, Callable
from http.client import IncompleteRead
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .base import Generation, ImageGen
from .xai_files import StoredXAIFile, XAIFileStore, cache_path_for_output

DEFAULT_BASE_URL = "https://api.x.ai/v1"
DEFAULT_MODEL = "grok-imagine-image-quality"
MAX_REFERENCE_IMAGES = 3
MAX_PROMPT_LENGTH = 8000
PROMPT_LENGTH_UNIT = "utf8_bytes"
DEFAULT_TIMEOUT_S = 300.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_S = 1.5
DEFAULT_USE_FILES_API = True
DEFAULT_FILE_RECOVERY_TIMEOUT_S = 300.0
DEFAULT_FILE_POLL_INTERVAL_S = 5.0


def _default_max_prompt_length(model: str) -> int | None:
    normalized = str(model or "").strip().lower()
    if normalized.startswith("grok-imagine-image"):
        return MAX_PROMPT_LENGTH
    return None


Requester = Callable[..., dict[str, Any]]
Downloader = Callable[[str, str], None]


class XAIReadTimeout(RuntimeError):
    pass


class XAIImageGen(ImageGen):
    name = "xai"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        base_url: str = DEFAULT_BASE_URL,
        api_key_env: str = "XAI_API_KEY",
        aspect_ratio: str = "16:9",
        resolution: str = "2k",
        cost_per_image: float = 0.07,
        cost_per_input_image: float = 0.01,
        pricing_source: str = "https://docs.x.ai/developers/models",
        pricing_as_of: str = "",
        max_reference_images: int = MAX_REFERENCE_IMAGES,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff_s: float = DEFAULT_RETRY_BACKOFF_S,
        use_files_api: bool = DEFAULT_USE_FILES_API,
        file_recovery_timeout_s: float = DEFAULT_FILE_RECOVERY_TIMEOUT_S,
        file_poll_interval_s: float = DEFAULT_FILE_POLL_INTERVAL_S,
        file_store_factory: Callable[[str, str], XAIFileStore] | None = None,
        requester: Requester | None = None,
        downloader: Downloader | None = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.aspect_ratio = aspect_ratio
        self.resolution = resolution
        self.cost_per_image = cost_per_image
        self.cost_per_input_image = cost_per_input_image
        self.pricing_source = str(pricing_source or "config")
        self.pricing_as_of = str(pricing_as_of or "")
        self.max_reference_images = max_reference_images
        self.max_prompt_length = _default_max_prompt_length(model)
        self.prompt_length_unit = PROMPT_LENGTH_UNIT
        self.timeout_s = float(timeout_s)
        self.max_retries = max(1, int(max_retries))
        self.retry_backoff_s = max(0.0, float(retry_backoff_s))
        self.use_files_api = bool(use_files_api)
        self.file_recovery_timeout_s = float(file_recovery_timeout_s)
        self.file_poll_interval_s = float(file_poll_interval_s)
        self._file_stores: dict[Path, XAIFileStore] = {}
        self._file_store_factory = (
            file_store_factory or self._default_file_store_factory
        )
        self._requester = requester or _request_json
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
        if len(refs) > self.max_reference_images:
            raise ValueError(
                f"{self.model} accepts at most {self.max_reference_images} reference "
                f"images, but {len(refs)} were passed."
            )
        missing = [ref for ref in refs if not Path(ref).is_file()]
        if missing:
            raise FileNotFoundError(f"xAI image reference file not found: {missing[0]}")

        api_key = os.environ.get(self.api_key_env)
        if not api_key and self._requester is _request_json:
            raise RuntimeError(f"{self.api_key_env} not set - add it to .env (see `cli doctor`).")

        aspect_ratio = kwargs.get("aspect_ratio") or self.aspect_ratio
        resolution = kwargs.get("resolution") or kwargs.get("image_resolution") or self.resolution
        store = (
            self._file_store_factory(out_path, api_key or "")
            if self.use_files_api
            else None
        )
        if self.use_files_api and store is None:
            raise RuntimeError(
                "xAI Files mode requires a file store, but file_store_factory "
                "returned None."
            )
        stored_refs: list[StoredXAIFile] = (
            [store.resolve_input(ref) for ref in refs]
            if store is not None
            else []
        )
        reference_hashes = [stored.content_sha256 for stored in stored_refs]
        reference_file_ids = [stored.file_id for stored in stored_refs]
        body: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt.strip(),
            "aspect_ratio": str(aspect_ratio),
            "resolution": str(resolution),
            "response_format": "url",
        }
        endpoint = "generations"
        if refs:
            endpoint = "edits"
        if stored_refs:
            items = [{"file_id": stored.file_id} for stored in stored_refs]
            if len(items) == 1:
                body["image"] = items[0]
            else:
                body["images"] = items
        elif refs and not self.use_files_api:
            encoded = [
                {"type": "image_url", "url": _data_url(ref)}
                for ref in refs
            ]
            if len(encoded) == 1:
                body["image"] = encoded[0]
            else:
                body["images"] = encoded

        generation_key = _generation_key(
            out_path=out_path,
            model=self.model,
            prompt=prompt,
            resolution=str(resolution),
            aspect_ratio=str(aspect_ratio),
            reference_hashes=reference_hashes,
            seed=kwargs.get("seed"),
        )
        estimated_cost = round(
            float(kwargs.get("cost_per_image", self.cost_per_image))
            + len(refs)
            * float(kwargs.get("cost_per_input_image", self.cost_per_input_image)),
            4,
        )
        output_filename = f"studio-output-{generation_key}.png"
        allow_ambiguous_resubmit = (
            kwargs.get("allow_ambiguous_resubmit") is True
        )
        attempt_count = 1
        cumulative_estimated_cost_usd = estimated_cost
        request_cost_usd = estimated_cost
        request_cost_provenance = "current_request"
        start = time.time()
        if store is not None:
            body["storage_options"] = {"filename": output_filename}
            prior = store.output_record(generation_key)
            prior_status = str((prior or {}).get("request_status") or "")
            prior_attempt_count = max(
                0, int((prior or {}).get("attempt_count") or 0)
            )
            prior_cumulative_cost = max(
                0.0,
                float(
                    (prior or {}).get("cumulative_estimated_cost_usd") or 0.0
                ),
            )
            if prior_status == "submitted":
                prior_attempt_count = max(1, prior_attempt_count)
                if prior_cumulative_cost <= 0:
                    prior_cumulative_cost = max(
                        0.0,
                        float(
                            (prior or {}).get("estimated_cost_usd")
                            or estimated_cost
                        ),
                    )
            prior_file_id = str((prior or {}).get("file_id") or "")
            if prior_file_id:
                existing = StoredXAIFile(
                    prior_file_id,
                    str((prior or {}).get("filename") or output_filename),
                )
            else:
                existing = store.find_by_filename(output_filename)
            if existing is not None:
                store.download(existing.file_id, out_path)
                store.mark_output_completed(
                    generation_key, file_id=existing.file_id
                )
                if prior_status == "completed":
                    recovered_cost = 0.0
                    recovered_cost_provenance = "previously_completed"
                    recovered_attempt_count = prior_attempt_count
                    recovered_cumulative_cost = prior_cumulative_cost
                elif prior_status == "submitted":
                    recovered_cost = prior_cumulative_cost
                    recovered_cost_provenance = "ambiguous_recovery"
                    recovered_attempt_count = prior_attempt_count
                    recovered_cumulative_cost = prior_cumulative_cost
                else:
                    recovered_cost = estimated_cost
                    recovered_cost_provenance = "recovered_unknown"
                    recovered_attempt_count = 1
                    recovered_cumulative_cost = estimated_cost
                result = self._generation_result(
                    out_path=out_path,
                    start=start,
                    cost_usd=recovered_cost,
                    endpoint=endpoint,
                    refs=refs,
                    reference_file_ids=reference_file_ids,
                    generation_key=generation_key,
                    stored_output_file_id=existing.file_id,
                    aspect_ratio=str(aspect_ratio),
                    resolution=str(resolution),
                    seed=kwargs.get("seed"),
                    recovered_before_request=True,
                    cost_provenance=recovered_cost_provenance,
                )
                result.meta.update({
                    "attempt_count": recovered_attempt_count,
                    "cumulative_estimated_cost_usd": round(
                        recovered_cumulative_cost, 4
                    ),
                })
                return result
            if prior_status == "submitted" and not allow_ambiguous_resubmit:
                raise RuntimeError(
                    f"Stored output {output_filename!r} is still unavailable for an "
                    "ambiguous prior submitted request; resume later to recover it, "
                    "or call generate(..., allow_ambiguous_resubmit=True) to "
                    "explicitly authorize one private Files-mode paid retry."
                )
            if prior_status == "submitted":
                attempt_count = prior_attempt_count + 1
                cumulative_estimated_cost_usd = round(
                    prior_cumulative_cost + estimated_cost, 4
                )
                request_cost_usd = cumulative_estimated_cost_usd
                request_cost_provenance = "ambiguous_resubmit"
            store.mark_output_submitted(
                generation_key,
                filename=output_filename,
                local_path=str(Path(out_path)),
                model=self.model,
                reference_hashes=reference_hashes,
                estimated_cost_usd=estimated_cost,
                attempt_count=attempt_count,
                cumulative_estimated_cost_usd=cumulative_estimated_cost_usd,
            )

        try:
            response = self._request_with_retry(
                f"{self.base_url}/images/{endpoint}",
                api_key=api_key or "",
                json_body=body,
            )
        except XAIReadTimeout as exc:
            if store is None:
                raise
            recovered = store.poll_for_filename(
                output_filename, self.file_recovery_timeout_s
            )
            if recovered is None:
                raise RuntimeError(
                    f"{exc} Stored output {output_filename!r} was not visible after "
                    f"{self.file_recovery_timeout_s:g}s; resume later to recover it. "
                    "If recovery remains unavailable, call generate(..., "
                    "allow_ambiguous_resubmit=True) to explicitly authorize one "
                    "private Files-mode paid retry."
                ) from exc
            store.download(recovered.file_id, out_path)
            store.mark_output_completed(
                generation_key, file_id=recovered.file_id
            )
            result = self._generation_result(
                out_path=out_path,
                start=start,
                cost_usd=request_cost_usd,
                endpoint=endpoint,
                refs=refs,
                reference_file_ids=reference_file_ids,
                generation_key=generation_key,
                stored_output_file_id=recovered.file_id,
                aspect_ratio=str(aspect_ratio),
                resolution=str(resolution),
                seed=kwargs.get("seed"),
                recovered_after_timeout=True,
                cost_provenance=(
                    request_cost_provenance
                    if request_cost_provenance == "ambiguous_resubmit"
                    else "timed_out_request"
                ),
            )
            result.meta.update({
                "attempt_count": attempt_count,
                "cumulative_estimated_cost_usd": round(
                    cumulative_estimated_cost_usd, 4
                ),
            })
            return result
        image_url = _image_url(response)
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._downloader(image_url, str(path))
        stored_output_file_id = ""
        if store is not None:
            stored_output_file_id = _file_output_id(response)
            store.mark_output_completed(
                generation_key, file_id=stored_output_file_id
            )
        result = self._generation_result(
            out_path=out_path,
            start=start,
            cost_usd=request_cost_usd,
            endpoint=endpoint,
            refs=refs,
            reference_file_ids=reference_file_ids,
            generation_key=generation_key,
            stored_output_file_id=stored_output_file_id,
            aspect_ratio=str(aspect_ratio),
            resolution=str(resolution),
            seed=kwargs.get("seed"),
            usage=dict(response.get("usage") or {}),
            image_url=image_url,
            cost_provenance=request_cost_provenance,
        )
        if store is not None:
            result.meta.update({
                "attempt_count": attempt_count,
                "cumulative_estimated_cost_usd": round(
                    cumulative_estimated_cost_usd, 4
                ),
            })
        return result

    def _default_file_store_factory(
        self, out_path: str, api_key: str
    ) -> XAIFileStore:
        cache_path = cache_path_for_output(out_path)
        if cache_path in self._file_stores:
            return self._file_stores[cache_path]
        store = XAIFileStore(
            base_url=self.base_url,
            api_key=api_key,
            cache_path=cache_path,
            timeout_s=self.timeout_s,
            poll_interval_s=self.file_poll_interval_s,
            max_retries=self.max_retries,
            retry_backoff_s=self.retry_backoff_s,
        )
        self._file_stores[cache_path] = store
        return store

    def _generation_result(
        self,
        *,
        out_path: str,
        start: float,
        cost_usd: float,
        endpoint: str,
        refs: list[str],
        reference_file_ids: list[str],
        generation_key: str,
        stored_output_file_id: str,
        aspect_ratio: str,
        resolution: str,
        seed: Any,
        usage: dict[str, Any] | None = None,
        image_url: str = "",
        recovered_before_request: bool = False,
        recovered_after_timeout: bool = False,
        cost_provenance: str = "current_request",
    ) -> Generation:
        per_request_cost = self.cost_per_image + len(refs) * self.cost_per_input_image
        billed_requests = (
            max(1, round(float(cost_usd) / per_request_cost))
            if float(cost_usd) > 0 and per_request_cost > 0
            else 0
        )
        cost_tracking = {
            "usage": {
                "output_images": billed_requests,
                "input_images": len(refs) * billed_requests,
            },
            "native_cost": round(float(cost_usd), 4),
            "native_currency": "USD",
            "estimate": True,
            "usage_missing": False,
            "pricing": {
                "model": self.model,
                "resolution": resolution,
                "output_image_usd": self.cost_per_image,
                "input_image_usd": self.cost_per_input_image,
                "source": self.pricing_source,
                "as_of": self.pricing_as_of,
            },
        }
        meta = {
            "endpoint": endpoint,
            "image_url": image_url,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "reference_images": refs,
            "reference_file_ids": reference_file_ids,
            "cost_per_input_image": self.cost_per_input_image,
            "usage": dict(usage or {}),
            "cost_tracking": cost_tracking,
            "seed": seed,
            "generation_key": generation_key,
            "stored_output_file_id": stored_output_file_id,
            "cost_provenance": cost_provenance,
        }
        if recovered_before_request:
            meta["recovered_before_request"] = True
        if recovered_after_timeout:
            meta["recovered_after_timeout"] = True
        return Generation(
            content=out_path,
            provider=self.name,
            model=self.model,
            cost_usd=round(float(cost_usd), 4),
            seconds=round(time.time() - start, 2),
            meta=meta,
        )

    def _request_with_retry(
        self,
        url: str,
        *,
        api_key: str,
        json_body: dict[str, Any],
    ) -> dict[str, Any]:
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._requester(
                    url,
                    api_key=api_key,
                    json_body=json_body,
                    timeout_s=self.timeout_s,
                )
            except HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"xAI Images API HTTP {exc.code}: {body}") from exc
            except TimeoutError as exc:
                raise _ambiguous_timeout(self.timeout_s) from exc
            except URLError as exc:
                reason = getattr(exc, "reason", exc)
                if _is_timeout(exc):
                    raise _ambiguous_timeout(self.timeout_s) from exc
                if _is_retryable_urlerror(exc):
                    if attempt < self.max_retries:
                        self._sleep_before_retry(attempt)
                        continue
                    raise RuntimeError(
                        f"xAI Images API upload/connection failed after {self.max_retries} "
                        f"attempts: {reason}. Check connectivity or raise image_timeout_s, "
                        "then resume."
                    ) from exc
                raise RuntimeError(f"xAI Images API request failed: {reason}") from exc
            except (ConnectionError, IncompleteRead) as exc:
                # A dropped/reset connection or truncated read is transient. RemoteDisconnected
                # ("Remote end closed connection without response") is a ConnectionResetError,
                # not a URLError, so it slips past the branch above — retry it with backoff.
                if attempt < self.max_retries:
                    self._sleep_before_retry(attempt)
                    continue
                raise RuntimeError(
                    f"xAI Images API upload/connection failed after {self.max_retries} "
                    f"attempts: {exc}. Check connectivity or raise image_timeout_s, then resume."
                ) from exc
        raise AssertionError("xAI image retry loop exited unexpectedly")

    def _sleep_before_retry(self, attempt: int) -> None:
        delay = self.retry_backoff_s * (2 ** (attempt - 1))
        if delay:
            time.sleep(delay)


def _data_url(path: str) -> str:
    file_path = Path(path)
    mime, _encoding = mimetypes.guess_type(file_path.name)
    if not mime or not mime.startswith("image/"):
        mime = "image/png"
    encoded = base64.b64encode(file_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _generation_key(
    *,
    out_path: str,
    model: str,
    prompt: str,
    resolution: str,
    aspect_ratio: str,
    reference_hashes: list[str],
    seed: Any,
) -> str:
    output = Path(out_path).resolve()
    project_id = ""
    relative_output = output.name
    for parent in (output.parent, *output.parents):
        project_file = parent / "project.json"
        if project_file.is_file():
            project_id = parent.name
            relative_output = output.relative_to(parent).as_posix()
            break
    payload = {
        "project_id": project_id,
        "output": relative_output,
        "model": model,
        "prompt": prompt.strip(),
        "resolution": str(resolution),
        "aspect_ratio": str(aspect_ratio),
        "reference_hashes": reference_hashes,
        "seed": seed,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_output_id(response: dict[str, Any]) -> str:
    data = response.get("data")
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        raise RuntimeError("xAI Images API returned no image data.")
    file_output = data[0].get("file_output")
    file_id = file_output.get("file_id") if isinstance(file_output, dict) else None
    if not file_id:
        raise RuntimeError(
            "xAI Images API did not return stored file_output.file_id"
        )
    return str(file_id)


def _image_url(response: dict[str, Any]) -> str:
    data = response.get("data") if isinstance(response, dict) else None
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        raise RuntimeError("xAI Images API returned no image data.")
    url = data[0].get("url")
    if not url:
        raise RuntimeError("xAI Images API response did not include data[0].url.")
    return str(url)


def _ambiguous_timeout(timeout_s: float) -> XAIReadTimeout:
    return XAIReadTimeout(
        f"xAI Images API timed out after {timeout_s:g}s waiting for the response; "
        "it was not retried to avoid duplicating a paid generation. Raise "
        "image_timeout_s and resume the project."
    )


def _is_timeout(error: object) -> bool:
    reason = getattr(error, "reason", None)
    if isinstance(error, TimeoutError) or isinstance(reason, TimeoutError):
        return True
    text = f"{error} {reason or ''}".lower()
    return "timed out" in text or "timeout" in text


def _is_retryable_urlerror(exc: URLError) -> bool:
    reason = getattr(exc, "reason", None)
    return (
        not _is_timeout(exc)
        and isinstance(reason, (socket.gaierror, ConnectionError))
    )


def _request_json(
    url: str,
    *,
    api_key: str,
    json_body: dict[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(json_body).encode("utf-8"),
        method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _download_url(url: str, out_path: str) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # xAI serves the generated image from its imgen.x.ai CDN, which rejects urllib's
    # default "Python-urllib/x.y" User-Agent with HTTP 403 Forbidden. Send a browser-like
    # User-Agent so the download succeeds (the URL itself is unauthenticated).
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 (studio-agent)"})
    with urlopen(request) as response, path.open("wb") as output:
        shutil.copyfileobj(response, output)
