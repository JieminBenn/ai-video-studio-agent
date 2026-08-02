import base64
import io
import json
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

from studio_agent.providers import xai_files
from studio_agent.providers.xai_files import XAIFileStore


PNG_A = b"\x89PNG\r\n\x1a\nreference-a"
PNG_B = b"\x89PNG\r\n\x1a\nreference-b"


class FakeFilesRemote:
    def __init__(self):
        self.files = {}
        self.uploads = []
        self.downloads = []
        self.next_id = 1

    def request(self, method, url, *, api_key, timeout_s):
        assert api_key == "secret"
        if method == "GET" and "/files/" in url:
            file_id = url.rsplit("/", 1)[-1]
            if file_id not in self.files:
                raise HTTPError(url, 404, "not found", {}, None)
            return dict(self.files[file_id])
        if method == "GET" and url.split("?", 1)[0].endswith("/files"):
            return {"data": list(self.files.values())}
        raise AssertionError((method, url))

    def upload(self, url, *, api_key, path, filename, timeout_s):
        file_id = f"file-{self.next_id}"
        self.next_id += 1
        row = {
            "id": file_id,
            "filename": filename,
            "bytes": Path(path).stat().st_size,
            "object": "file",
            "purpose": "assistants",
        }
        self.files[file_id] = row
        self.uploads.append((str(path), filename))
        return dict(row)

    def download(self, url, *, api_key, out_path, timeout_s):
        file_id = url.split("/files/", 1)[1].split("/", 1)[0]
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(f"download:{file_id}".encode())
        self.downloads.append(file_id)


def _store(tmp_path, remote):
    return XAIFileStore(
        base_url="https://api.x.ai/v1",
        api_key="secret",
        cache_path=tmp_path / "provider_cache" / "xai_files.json",
        timeout_s=30,
        poll_interval_s=0,
        requester=remote.request,
        uploader=remote.upload,
        downloader=remote.download,
        sleeper=lambda _seconds: None,
    )


def test_resolve_input_uploads_once_and_persists_hash_cache(tmp_path):
    remote = FakeFilesRemote()
    image = tmp_path / "hero.png"
    image.write_bytes(PNG_A)

    first = _store(tmp_path, remote).resolve_input(str(image))
    second = _store(tmp_path, remote).resolve_input(str(image))

    assert first.file_id == second.file_id == "file-1"
    assert len(remote.uploads) == 1
    manifest = json.loads((tmp_path / "provider_cache/xai_files.json").read_text())
    assert manifest["version"] == 1
    assert list(manifest["inputs"].values())[0]["file_id"] == "file-1"
    assert "secret" not in json.dumps(manifest)


def test_changed_bytes_at_same_path_create_a_new_remote_input(tmp_path):
    remote = FakeFilesRemote()
    image = tmp_path / "hero.png"
    image.write_bytes(PNG_A)
    first = _store(tmp_path, remote).resolve_input(str(image))
    image.write_bytes(PNG_B)

    second = _store(tmp_path, remote).resolve_input(str(image))

    assert first.file_id == "file-1"
    assert second.file_id == "file-2"
    assert len(remote.uploads) == 2


def test_deleted_cached_id_reuploads_the_authoritative_local_file(tmp_path):
    remote = FakeFilesRemote()
    image = tmp_path / "hero.png"
    image.write_bytes(PNG_A)
    first = _store(tmp_path, remote).resolve_input(str(image))
    remote.files.pop(first.file_id)

    replacement = _store(tmp_path, remote).resolve_input(str(image))

    assert replacement.file_id == "file-2"
    assert len(remote.uploads) == 2


def test_missing_manifest_recovers_deterministic_remote_filename(tmp_path):
    remote = FakeFilesRemote()
    image = tmp_path / "hero.png"
    image.write_bytes(PNG_A)
    first_store = _store(tmp_path, remote)
    first = first_store.resolve_input(str(image))
    first_store.cache_path.unlink()

    recovered = _store(tmp_path, remote).resolve_input(str(image))

    assert recovered.file_id == first.file_id
    assert len(remote.uploads) == 1


def test_output_rows_and_authenticated_download_are_plain_file_state(tmp_path):
    remote = FakeFilesRemote()
    image = tmp_path / "hero.png"
    image.write_bytes(PNG_A)
    stored = _store(tmp_path, remote).resolve_input(str(image))
    store = _store(tmp_path, remote)
    store.mark_output_submitted(
        "key-1",
        filename="studio-output-key-1.png",
        local_path="bible/reference.png",
        model="grok-imagine-image-quality",
        reference_hashes=[stored.content_sha256],
        estimated_cost_usd=0.08,
    )
    store.mark_output_completed("key-1", file_id=stored.file_id)
    out = tmp_path / "recovered.png"

    store.download(stored.file_id, str(out))

    assert out.read_bytes() == b"download:file-1"
    assert store.output_record("key-1")["request_status"] == "completed"


def test_poll_finds_output_without_resubmission(tmp_path):
    remote = FakeFilesRemote()
    sleeps = []

    def sleeper(seconds):
        sleeps.append(seconds)
        if not remote.files:
            remote.files["file-output"] = {
                "id": "file-output",
                "filename": "studio-output-key.png",
                "bytes": 20,
            }

    store = XAIFileStore(
        base_url="https://api.x.ai/v1",
        api_key="secret",
        cache_path=tmp_path / "cache.json",
        timeout_s=30,
        poll_interval_s=0.01,
        requester=remote.request,
        uploader=remote.upload,
        downloader=remote.download,
        sleeper=sleeper,
    )

    found = store.poll_for_filename("studio-output-key.png", timeout_s=1)

    assert found.file_id == "file-output"
    assert sleeps


def test_corrupt_manifest_fails_with_cache_path(tmp_path):
    remote = FakeFilesRemote()
    cache = tmp_path / "provider_cache/xai_files.json"
    cache.parent.mkdir(parents=True)
    cache.write_text("not json")
    image = tmp_path / "hero.png"
    image.write_bytes(PNG_A)

    with pytest.raises(ValueError, match="xai_files.json"):
        _store(tmp_path, remote).resolve_input(str(image))


def test_ambiguous_upload_timeout_recovers_by_deterministic_filename(tmp_path):
    remote = FakeFilesRemote()
    image = tmp_path / "hero.png"
    image.write_bytes(PNG_A)
    upload_calls = []

    def upload_then_timeout(url, *, api_key, path, filename, timeout_s):
        upload_calls.append(filename)
        row = remote.upload(
            url,
            api_key=api_key,
            path=path,
            filename=filename,
            timeout_s=timeout_s,
        )
        assert row["id"] == "file-1"
        raise TimeoutError("The read operation timed out")

    store = XAIFileStore(
        base_url="https://api.x.ai/v1",
        api_key="secret",
        cache_path=tmp_path / "provider_cache/xai_files.json",
        timeout_s=30,
        poll_interval_s=0,
        retry_backoff_s=0,
        requester=remote.request,
        uploader=upload_then_timeout,
        downloader=remote.download,
        sleeper=lambda _seconds: None,
    )

    resolved = store.resolve_input(str(image))

    assert resolved.file_id == "file-1"
    assert len(upload_calls) == 1


def test_concrete_upload_transport_sends_private_raw_multipart_file(
    tmp_path, monkeypatch
):
    image = tmp_path / "hero.png"
    image.write_bytes(PNG_A)
    captured = {}

    def fake_urlopen(request, *, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return io.BytesIO(
            json.dumps({
                "id": "file-transport",
                "filename": "stable-hero.png",
                "bytes": len(PNG_A),
            }).encode()
        )

    monkeypatch.setattr(xai_files, "urlopen", fake_urlopen)

    response = xai_files._upload_file(
        "https://api.x.ai/v1/files",
        api_key="xai-secret",
        path=image,
        filename="stable-hero.png",
        timeout_s=27,
    )

    request = captured["request"]
    body = request.data
    content_type = request.get_header("Content-type")
    boundary = content_type.split("boundary=", 1)[1].encode()
    file_headers, file_part = body.split(
        b'Content-Disposition: form-data; name="file"; ', 1
    )
    raw_payload = file_part.split(b"\r\n\r\n", 1)[1].split(
        b"\r\n--" + boundary + b"--\r\n", 1
    )[0]

    assert response["id"] == "file-transport"
    assert captured["timeout"] == 27
    assert request.get_header("Authorization") == "Bearer xai-secret"
    assert request.get_method() == "POST"
    assert content_type.startswith("multipart/form-data; boundary=")
    assert b'name="purpose"' in file_headers
    assert b"assistants" in file_headers
    assert b'filename="stable-hero.png"' in file_part
    assert raw_payload == PNG_A
    assert base64.b64encode(PNG_A) not in body
    assert b"data:image" not in body
    assert b"public" not in body.lower()
    assert b"expiry" not in body.lower()
    assert b"expires" not in body.lower()
