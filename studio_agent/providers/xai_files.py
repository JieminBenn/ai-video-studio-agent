from __future__ import annotations

import hashlib
import json
import mimetypes
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


JsonRequester = Callable[..., dict[str, Any]]
Uploader = Callable[..., dict[str, Any]]
Downloader = Callable[..., None]


@dataclass(frozen=True)
class StoredXAIFile:
    file_id: str
    filename: str
    content_sha256: str = ""
    bytes: int = 0


def cache_path_for_output(out_path: str) -> Path:
    output = Path(out_path).resolve()
    for parent in (output.parent, *output.parents):
        if (parent / "project.json").is_file():
            return parent / "provider_cache" / "xai_files.json"
    return output.parent / ".xai_files.json"


class XAIFileStore:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        cache_path: str | Path,
        timeout_s: float,
        poll_interval_s: float,
        max_retries: int = 3,
        retry_backoff_s: float = 1.5,
        requester: JsonRequester | None = None,
        uploader: Uploader | None = None,
        downloader: Downloader | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.cache_path = Path(cache_path)
        self.timeout_s = float(timeout_s)
        self.poll_interval_s = max(0.0, float(poll_interval_s))
        self.max_retries = max(1, int(max_retries))
        self.retry_backoff_s = max(0.0, float(retry_backoff_s))
        self._requester = requester or _request_json
        self._uploader = uploader or _upload_file
        self._downloader = downloader or _download_file
        self._sleep = sleeper
        self._validated_ids: set[str] = set()

    def resolve_input(self, path: str) -> StoredXAIFile:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"xAI reference file not found: {source}")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        manifest = self._load()
        cached = manifest["inputs"].get(digest)
        if cached and self._valid(str(cached.get("file_id") or "")):
            return StoredXAIFile(
                file_id=str(cached["file_id"]),
                filename=str(cached["filename"]),
                content_sha256=digest,
                bytes=int(cached.get("bytes") or source.stat().st_size),
            )
        suffix = source.suffix.lower() or mimetypes.guess_extension(
            mimetypes.guess_type(source.name)[0] or "image/png"
        ) or ".png"
        filename = f"studio-ref-{digest[:24]}{suffix}"
        stored = self.find_by_filename(filename)
        if stored is None:
            stored = self._upload_with_recovery(source, filename, digest)
        manifest = self._load()
        manifest["inputs"][digest] = {
            "file_id": stored.file_id,
            "filename": stored.filename,
            "bytes": stored.bytes or source.stat().st_size,
            "local_paths": sorted(
                set(list((cached or {}).get("local_paths") or []) + [str(source)])
            ),
        }
        self._write(manifest)
        self._validated_ids.add(stored.file_id)
        return StoredXAIFile(
            stored.file_id,
            stored.filename,
            digest,
            stored.bytes or source.stat().st_size,
        )

    def find_by_filename(self, filename: str) -> StoredXAIFile | None:
        token = ""
        while True:
            params = {"limit": 100, "order": "desc", "sort_by": "created_at"}
            if token:
                params["pagination_token"] = token
            page = self._requester(
                "GET",
                f"{self.base_url}/files?{urlencode(params)}",
                api_key=self.api_key,
                timeout_s=self.timeout_s,
            )
            rows = list(page.get("data") or [])
            for row in rows:
                if row.get("filename") == filename:
                    return _stored_file(row)
            if len(rows) < 100:
                return None
            token = str(page.get("pagination_token") or "")
            if not token:
                return None

    def output_record(self, key: str) -> dict[str, Any] | None:
        row = self._load()["outputs"].get(key)
        return dict(row) if isinstance(row, dict) else None

    def mark_output_submitted(self, key: str, **row: Any) -> None:
        manifest = self._load()
        manifest["outputs"][key] = {**row, "request_status": "submitted"}
        self._write(manifest)

    def mark_output_completed(self, key: str, *, file_id: str) -> None:
        manifest = self._load()
        row = dict(manifest["outputs"].get(key) or {})
        row.update({"file_id": file_id, "request_status": "completed"})
        manifest["outputs"][key] = row
        self._write(manifest)

    def download(self, file_id: str, out_path: str) -> None:
        self._downloader(
            f"{self.base_url}/files/{file_id}/content",
            api_key=self.api_key,
            out_path=out_path,
            timeout_s=self.timeout_s,
        )

    def poll_for_filename(self, filename: str, timeout_s: float) -> StoredXAIFile | None:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while True:
            found = self.find_by_filename(filename)
            if found is not None:
                return found
            if time.monotonic() >= deadline:
                return None
            self._sleep(
                min(
                    self.poll_interval_s,
                    max(0.0, deadline - time.monotonic()),
                )
            )

    def _valid(self, file_id: str) -> bool:
        if not file_id:
            return False
        if file_id in self._validated_ids:
            return True
        try:
            self._requester(
                "GET",
                f"{self.base_url}/files/{file_id}",
                api_key=self.api_key,
                timeout_s=self.timeout_s,
            )
        except HTTPError as exc:
            if exc.code == 404:
                return False
            raise
        self._validated_ids.add(file_id)
        return True

    def _upload_with_recovery(
        self, source: Path, filename: str, digest: str
    ) -> StoredXAIFile:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                row = self._uploader(
                    f"{self.base_url}/files",
                    api_key=self.api_key,
                    path=source,
                    filename=filename,
                    timeout_s=self.timeout_s,
                )
                return _stored_file(row, content_sha256=digest)
            except HTTPError:
                raise
            except (TimeoutError, URLError) as exc:
                last_error = exc
                found = self.find_by_filename(filename)
                if found is not None:
                    return StoredXAIFile(
                        found.file_id,
                        found.filename,
                        digest,
                        found.bytes,
                    )
                if attempt < self.max_retries:
                    self._sleep(self.retry_backoff_s * attempt)
        raise RuntimeError(
            f"xAI file upload failed after {self.max_retries} attempts: {last_error}"
        ) from last_error

    def _load(self) -> dict[str, Any]:
        if not self.cache_path.is_file():
            return {"version": 1, "inputs": {}, "outputs": {}}
        try:
            value = json.loads(self.cache_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid xAI file cache: {self.cache_path}") from exc
        if value.get("version") != 1:
            raise ValueError(f"unsupported xAI file cache version: {self.cache_path}")
        value.setdefault("inputs", {})
        value.setdefault("outputs", {})
        return value

    def _write(self, value: dict[str, Any]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_name(
            f".{self.cache_path.name}.{uuid.uuid4().hex}.tmp"
        )
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(self.cache_path)


def _stored_file(
    row: dict[str, Any], *, content_sha256: str = ""
) -> StoredXAIFile:
    file_id = str(row.get("id") or row.get("file_id") or "")
    filename = str(row.get("filename") or "")
    if not file_id or not filename:
        raise RuntimeError("xAI Files API response needs id and filename")
    return StoredXAIFile(
        file_id=file_id,
        filename=filename,
        content_sha256=content_sha256,
        bytes=int(row.get("bytes") or row.get("size") or 0),
    )


def _request_json(
    method: str,
    url: str,
    *,
    api_key: str,
    timeout_s: float,
) -> dict[str, Any]:
    request = Request(
        url,
        method=method,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _upload_file(
    url: str,
    *,
    api_key: str,
    path: Path,
    filename: str,
    timeout_s: float,
) -> dict[str, Any]:
    boundary = f"studio-agent-{uuid.uuid4().hex}"
    mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            b'Content-Disposition: form-data; name="purpose"\r\n\r\n',
            b"assistants\r\n",
            f"--{boundary}\r\n".encode(),
            (
                'Content-Disposition: form-data; name="file"; '
                f'filename="{filename}"\r\n'
            ).encode(),
            f"Content-Type: {mime}\r\n\r\n".encode(),
            path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    request = Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    with urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _download_file(
    url: str,
    *,
    api_key: str,
    out_path: str,
    timeout_s: float,
) -> None:
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    request = Request(url, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with (
            urlopen(request, timeout=timeout_s) as response,
            temporary.open("wb") as handle,
        ):
            shutil.copyfileobj(response, handle)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
