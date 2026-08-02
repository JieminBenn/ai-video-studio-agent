import io
import json
import socket
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

from studio_agent.cli import build_providers
from studio_agent.providers import xai_image
from studio_agent.providers.xai_files import StoredXAIFile
from studio_agent.providers.xai_image import (
    XAIImageGen,
    _download_url,
    _generation_key,
)


PNG = b"\x89PNG\r\n\x1a\nxai-image"


def _write_test_png(_url, out_path):
    with open(out_path, "wb") as handle:
        handle.write(PNG)


def test_build_providers_selects_xai_image_lazily(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    providers = build_providers({
        "llm": "fake",
        "image": "xai",
        "image_model": "grok-imagine-image-quality",
        "video": "fake",
        "vlm": "fake",
        "tts": "fake",
        "music": "fake",
    })

    assert isinstance(providers.image, XAIImageGen)
    assert providers.image.model == "grok-imagine-image-quality"
    assert providers.image.timeout_s == 300.0
    assert providers.image.max_retries == 3
    assert providers.image.retry_backoff_s == 1.5
    assert providers.image.use_files_api is True
    assert providers.image.file_recovery_timeout_s == 300.0
    assert providers.image.file_poll_interval_s == 5.0


def test_build_providers_honors_xai_transport_overrides(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    providers = build_providers({
        "llm": "fake",
        "image": "xai",
        "image_model": "grok-imagine-image-quality",
        "image_timeout_s": 420.0,
        "image_max_retries": 5,
        "image_retry_backoff_s": 0.25,
        "xai_use_files_api": False,
        "xai_file_recovery_timeout_s": 90.0,
        "xai_file_poll_interval_s": 0.25,
        "video": "fake",
        "vlm": "fake",
        "tts": "fake",
        "music": "fake",
    })

    assert providers.image.timeout_s == 420.0
    assert providers.image.max_retries == 5
    assert providers.image.retry_backoff_s == 0.25
    assert providers.image.use_files_api is False
    assert providers.image.file_recovery_timeout_s == 90.0
    assert providers.image.file_poll_interval_s == 0.25


class FakeFileStore:
    def __init__(self):
        self.inputs = []
        self.outputs = {}
        self.downloaded = []

    def resolve_input(self, path):
        index = len(self.inputs) + 1
        self.inputs.append(path)
        return StoredXAIFile(
            file_id=f"file-ref-{index}",
            filename=f"ref-{index}.png",
            content_sha256=f"hash-{index}",
            bytes=10,
        )

    def find_by_filename(self, filename):
        return None

    def output_record(self, key):
        return self.outputs.get(key)

    def mark_output_submitted(self, key, **row):
        self.outputs[key] = {**row, "request_status": "submitted"}

    def mark_output_completed(self, key, *, file_id):
        self.outputs.setdefault(key, {}).update(
            {"file_id": file_id, "request_status": "completed"}
        )

    def poll_for_filename(self, filename, timeout_s):
        return None

    def download(self, file_id, out_path):
        Path(out_path).write_bytes(PNG)
        self.downloaded.append((file_id, out_path))


class FalseyFileStore(FakeFileStore):
    def __bool__(self):
        return False


def test_xai_files_mode_uses_falsey_store_without_base64_fallback(
    tmp_path, monkeypatch
):
    calls = []
    store = FalseyFileStore()

    def request(url, *, api_key, json_body, timeout_s):
        calls.append(json_body)
        return {
            "data": [{
                "url": "https://x.ai/out.png",
                "file_output": {"file_id": "file-output-falsey"},
            }],
        }

    ref = tmp_path / "ref.png"
    ref.write_bytes(PNG)
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    result = XAIImageGen(
        requester=request,
        downloader=_write_test_png,
        file_store_factory=lambda _out_path, _api_key: store,
    ).generate(
        "keep this private",
        out_path=str(tmp_path / "out.png"),
        reference_images=[str(ref)],
    )

    assert calls[0]["image"] == {"file_id": "file-ref-1"}
    assert "data:image" not in json.dumps(calls[0])
    assert calls[0]["storage_options"].keys() == {"filename"}
    generation_key = result.meta["generation_key"]
    assert store.outputs[generation_key]["request_status"] == "completed"
    assert store.outputs[generation_key]["file_id"] == "file-output-falsey"


def test_xai_files_mode_rejects_none_file_store_before_request(
    tmp_path, monkeypatch
):
    calls = []

    def request(url, *, api_key, json_body, timeout_s):
        calls.append(json_body)
        return {"data": [{"url": "https://x.ai/out.png"}]}

    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    gen = XAIImageGen(
        requester=request,
        downloader=_write_test_png,
        file_store_factory=lambda _out_path, _api_key: None,
    )

    with pytest.raises(RuntimeError, match="file_store_factory returned None"):
        gen.generate("must stay private", out_path=str(tmp_path / "out.png"))

    assert calls == []


def test_xai_image_generation_and_multi_reference_edit_use_private_files(
    tmp_path, monkeypatch
):
    calls = []
    store = FakeFileStore()

    def request(url, *, api_key, json_body, timeout_s):
        calls.append((url, api_key, json_body))
        return {
            "data": [{
                "url": "https://x.ai/out.png",
                "file_output": {"file_id": f"file-output-{len(calls)}"},
            }],
            "usage": {"cost_in_usd_ticks": 200000000},
        }

    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    gen = XAIImageGen(
        requester=request,
        downloader=_write_test_png,
        file_store_factory=lambda _out_path, _api_key: store,
    )
    refs = []
    for index in range(3):
        path = tmp_path / f"ref-{index}.png"
        path.write_bytes(PNG + bytes([index]))
        refs.append(str(path))

    result = gen.generate(
        "keep all three identities",
        out_path=str(tmp_path / "generated.png"),
        reference_images=refs,
        seed=7,
    )

    body = calls[0][2]
    assert calls[0][0].endswith("/images/edits")
    assert body["images"] == [
        {"file_id": "file-ref-1"},
        {"file_id": "file-ref-2"},
        {"file_id": "file-ref-3"},
    ]
    assert "data:image" not in json.dumps(body)
    assert body["storage_options"]["filename"].startswith("studio-output-")
    assert body["storage_options"].keys() == {"filename"}
    assert result.meta["reference_images"] == refs
    assert result.meta["reference_file_ids"] == [
        "file-ref-1", "file-ref-2", "file-ref-3"
    ]
    assert result.meta["stored_output_file_id"] == "file-output-1"
    assert result.cost_usd == 0.10
    assert result.meta["cost_tracking"]["usage"] == {
        "output_images": 1,
        "input_images": 3,
    }
    assert result.meta["cost_tracking"]["pricing"]["output_image_usd"] == 0.07
    assert result.meta["cost_tracking"]["pricing"]["input_image_usd"] == 0.01


def test_xai_files_mode_can_be_explicitly_disabled_for_diagnostics(
    tmp_path, monkeypatch
):
    calls = []

    def request(url, *, api_key, json_body, timeout_s):
        calls.append(json_body)
        return {"data": [{"url": "https://x.ai/out.png"}]}

    ref = tmp_path / "ref.png"
    ref.write_bytes(PNG)
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    XAIImageGen(
        requester=request,
        downloader=_write_test_png,
        use_files_api=False,
    ).generate(
        "diagnose",
        out_path=str(tmp_path / "out.png"),
        reference_images=[str(ref)],
    )

    assert calls[0]["image"]["url"].startswith("data:image/png;base64,")
    assert "storage_options" not in calls[0]


def test_download_url_sends_browser_user_agent(tmp_path, monkeypatch):
    # imgen.x.ai (xAI's CDN that serves returned image URLs) rejects urllib's default
    # "Python-urllib/x.y" User-Agent with HTTP 403, so the download must send a real one.
    captured = {}

    class _Resp:
        def __enter__(self):
            return io.BytesIO(PNG)

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, *args, **kwargs):
        captured["user_agent"] = request.get_header("User-agent")
        return _Resp()

    monkeypatch.setattr(xai_image, "urlopen", fake_urlopen)
    out = tmp_path / "img.png"
    _download_url("https://imgen.x.ai/out.png", str(out))

    assert captured["user_agent"], "download must set a User-Agent header"
    assert "python-urllib" not in captured["user_agent"].lower()
    assert out.read_bytes() == PNG


def test_xai_image_rejects_more_than_three_references(tmp_path):
    gen = XAIImageGen()
    with pytest.raises(ValueError, match="at most 3"):
        gen.generate("p", out_path=str(tmp_path / "o.png"), reference_images=["a", "b", "c", "d"])


def test_capabilities_report_max_reference_images():
    from studio_agent.providers.fake import FakeImageGen
    assert XAIImageGen().capabilities.max_reference_images == 3
    assert FakeImageGen().capabilities.max_reference_images >= 4


def test_xai_capabilities_report_utf8_prompt_limit():
    capabilities = XAIImageGen().capabilities

    assert capabilities.max_prompt_length == 8000
    assert capabilities.prompt_length_unit == "utf8_bytes"


def test_unverified_xai_image_model_has_no_assumed_prompt_limit():
    capabilities = XAIImageGen(model="future-xai-image-model").capabilities

    assert capabilities.max_prompt_length is None


def test_wrapped_timeout_polls_files_once_then_requires_actionable_resume(
    tmp_path, monkeypatch
):
    attempts = []
    store = FakeFileStore()

    def request(url, *, api_key, json_body, timeout_s):
        attempts.append(1)
        raise URLError(TimeoutError("The write operation timed out"))

    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    gen = XAIImageGen(
        requester=request,
        downloader=_write_test_png,
        file_store_factory=lambda _out_path, _api_key: store,
        max_retries=3,
        retry_backoff_s=0.0,
        file_recovery_timeout_s=0,
    )
    with pytest.raises(RuntimeError, match="resume later") as error:
        gen.generate("ambiguous wrapped timeout", out_path=str(tmp_path / "out.png"))

    assert len(attempts) == 1
    assert "allow_ambiguous_resubmit=True" in str(error.value)
    generation_key = next(iter(store.outputs))
    assert store.outputs[generation_key]["attempt_count"] == 1
    assert store.outputs[generation_key]["cumulative_estimated_cost_usd"] == 0.07


def test_transient_dns_failure_retries_then_succeeds(tmp_path, monkeypatch):
    attempts = []

    def request(url, *, api_key, json_body, timeout_s):
        attempts.append(1)
        if len(attempts) == 1:
            raise URLError(socket.gaierror(8, "host not found"))
        return {"data": [{"url": "https://x.ai/recovered.png"}]}

    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    gen = XAIImageGen(
        requester=request,
        downloader=_write_test_png,
        max_retries=3,
        retry_backoff_s=0.0,
        use_files_api=False,
    )
    gen.generate("recover DNS", out_path=str(tmp_path / "out.png"))

    assert len(attempts) == 2


def test_connection_reset_retries_then_succeeds(tmp_path, monkeypatch):
    # http.client.RemoteDisconnected ("Remote end closed connection without response") is a
    # ConnectionResetError, not a URLError, so the old retry classifier missed it and a
    # transient drop aborted the bible/image stage with the raw message.
    attempts = []

    def request(url, *, api_key, json_body, timeout_s):
        attempts.append(1)
        if len(attempts) == 1:
            raise ConnectionResetError("Remote end closed connection without response")
        return {"data": [{"url": "https://x.ai/recovered.png"}]}

    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    gen = XAIImageGen(
        requester=request,
        downloader=_write_test_png,
        max_retries=3,
        retry_backoff_s=0.0,
        use_files_api=False,
    )
    gen.generate("recover reset", out_path=str(tmp_path / "out.png"))

    assert len(attempts) == 2


def test_generic_timeout_recovers_stored_output_without_second_paid_request(
    tmp_path, monkeypatch
):
    attempts = []
    store = FakeFileStore()
    recovered = StoredXAIFile(
        "file-recovered", "studio-output-recovered.png", bytes=20
    )

    def request(url, *, api_key, json_body, timeout_s):
        attempts.append(json_body)
        raise TimeoutError("timed out")

    def poll(filename, timeout_s):
        assert filename.startswith("studio-output-")
        assert timeout_s == 45
        return recovered

    store.poll_for_filename = poll
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    result = XAIImageGen(
        requester=request,
        downloader=_write_test_png,
        file_store_factory=lambda _out_path, _api_key: store,
        file_recovery_timeout_s=45,
        retry_backoff_s=0,
    ).generate("slow response", out_path=str(tmp_path / "out.png"))

    assert len(attempts) == 1
    assert store.downloaded[0][0] == "file-recovered"
    assert result.meta["recovered_after_timeout"] is True
    assert result.cost_usd == 0.07
    assert result.meta["attempt_count"] == 1
    assert result.meta["cumulative_estimated_cost_usd"] == 0.07


def test_ambiguous_timeout_requires_explicit_private_resubmit_authorization(
    tmp_path, monkeypatch
):
    paid_requests = []
    store = FakeFileStore()
    reference = tmp_path / "reference.png"
    reference.write_bytes(PNG)
    store.resolve_input = lambda path: StoredXAIFile(
        "file-private-ref",
        "private-ref.png",
        content_sha256="stable-private-hash",
        bytes=len(PNG),
    )

    def request(url, *, api_key, json_body, timeout_s):
        paid_requests.append(json_body)
        if len(paid_requests) == 1:
            raise TimeoutError("The read operation timed out")
        return {
            "data": [{
                "url": "https://x.ai/retried.png",
                "file_output": {"file_id": "file-authorized-output"},
            }],
            "usage": {"cost_in_usd_ticks": 200000000},
        }

    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    gen = XAIImageGen(
        requester=request,
        downloader=_write_test_png,
        file_store_factory=lambda _out_path, _api_key: store,
        file_recovery_timeout_s=0,
        retry_backoff_s=0,
    )
    output = str(tmp_path / "out.png")

    with pytest.raises(RuntimeError, match="resume later") as timeout_error:
        gen.generate(
            "ambiguous timeout",
            out_path=output,
            reference_images=[str(reference)],
        )
    assert len(paid_requests) == 1
    assert "allow_ambiguous_resubmit=True" in str(timeout_error.value)
    generation_key = next(iter(store.outputs))
    first_attempt = store.outputs[generation_key]
    assert first_attempt["attempt_count"] == 1
    assert first_attempt["cumulative_estimated_cost_usd"] == 0.08

    with pytest.raises(RuntimeError) as resume_error:
        gen.generate(
            "ambiguous timeout",
            out_path=output,
            reference_images=[str(reference)],
        )
    assert len(paid_requests) == 1
    assert "allow_ambiguous_resubmit=True" in str(resume_error.value)
    assert store.outputs[generation_key] == first_attempt

    result = gen.generate(
        "ambiguous timeout",
        out_path=output,
        reference_images=[str(reference)],
        allow_ambiguous_resubmit=True,
    )

    assert len(paid_requests) == 2
    retry_body = paid_requests[1]
    assert retry_body["image"] == {"file_id": "file-private-ref"}
    assert retry_body["storage_options"]["filename"].startswith(
        "studio-output-"
    )
    assert "data:image" not in json.dumps(retry_body)
    assert result.meta["stored_output_file_id"] == "file-authorized-output"
    assert store.outputs[generation_key]["attempt_count"] == 2
    assert store.outputs[generation_key]["cumulative_estimated_cost_usd"] == 0.16
    assert result.cost_usd == 0.16
    assert result.meta["cost_provenance"] == "ambiguous_resubmit"
    assert result.meta["attempt_count"] == 2
    assert result.meta["cumulative_estimated_cost_usd"] == 0.16


def test_legacy_submitted_row_recovery_uses_stored_conservative_cost(
    tmp_path, monkeypatch
):
    store = FakeFileStore()
    store.output_record = lambda key: {
        "filename": f"studio-output-{key}.png",
        "request_status": "submitted",
        "estimated_cost_usd": 0.11,
    }
    store.find_by_filename = lambda filename: StoredXAIFile(
        "file-legacy-output", filename, bytes=20
    )
    monkeypatch.setenv("XAI_API_KEY", "xai-test")

    result = XAIImageGen(
        requester=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("paid generation must not run")
        ),
        downloader=_write_test_png,
        file_store_factory=lambda _out_path, _api_key: store,
    ).generate("legacy resume", out_path=str(tmp_path / "out.png"))

    assert result.cost_usd == 0.11
    assert result.meta["cost_provenance"] == "ambiguous_recovery"
    assert result.meta["attempt_count"] == 1
    assert result.meta["cumulative_estimated_cost_usd"] == 0.11


def test_resume_recovers_remote_output_before_paid_generation(tmp_path, monkeypatch):
    calls = []
    store = FakeFileStore()
    store.find_by_filename = lambda filename: StoredXAIFile(
        "file-existing", filename, bytes=20
    )

    def request(url, *, api_key, json_body, timeout_s):
        calls.append(json_body)
        raise AssertionError("paid generation must not run")

    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    result = XAIImageGen(
        requester=request,
        downloader=_write_test_png,
        file_store_factory=lambda _out_path, _api_key: store,
    ).generate("resume", out_path=str(tmp_path / "out.png"))

    assert calls == []
    assert result.meta["recovered_before_request"] is True
    assert result.meta["stored_output_file_id"] == "file-existing"
    assert result.cost_usd == 0.07


def test_resume_of_known_completed_output_adds_no_new_generation_cost(
    tmp_path, monkeypatch
):
    store = FakeFileStore()
    store.output_record = lambda key: {
        "file_id": "file-completed",
        "filename": f"studio-output-{key}.png",
        "request_status": "completed",
        "estimated_cost_usd": 0.07,
    }
    monkeypatch.setenv("XAI_API_KEY", "xai-test")

    result = XAIImageGen(
        requester=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("paid generation must not run")
        ),
        downloader=_write_test_png,
        file_store_factory=lambda _out_path, _api_key: store,
    ).generate("resume completed", out_path=str(tmp_path / "out.png"))

    assert result.meta["cost_provenance"] == "previously_completed"
    assert result.cost_usd == 0.0


def test_generic_timeout_without_files_mode_is_terminal_without_retry(
    tmp_path, monkeypatch
):
    attempts = []

    def request(url, *, api_key, json_body, timeout_s):
        attempts.append(1)
        raise TimeoutError("timed out")

    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    gen = XAIImageGen(
        requester=request,
        downloader=_write_test_png,
        max_retries=3,
        retry_backoff_s=0.0,
        use_files_api=False,
    )
    with pytest.raises(RuntimeError, match="not retried to avoid duplicating a paid generation"):
        gen.generate("slow response", out_path=str(tmp_path / "out.png"))

    assert len(attempts) == 1


def test_generation_key_is_repeatable_and_changes_with_meaningful_input(tmp_path):
    inputs = {
        "out_path": str(tmp_path / "keyframe.png"),
        "model": "grok-imagine-image-quality",
        "prompt": "  a lighthouse at blue hour  ",
        "resolution": "2k",
        "aspect_ratio": "16:9",
        "reference_hashes": ["hero-hash", "location-hash"],
        "seed": 7,
    }

    first = _generation_key(**inputs)
    equivalent = _generation_key(**dict(inputs))
    changed = _generation_key(**{**inputs, "prompt": "a lighthouse at dawn"})

    assert first == equivalent
    assert first != changed


@pytest.mark.parametrize("status", [400, 503])
def test_http_responses_are_terminal_without_retry(tmp_path, monkeypatch, status):
    attempts = []

    def request(url, *, api_key, json_body, timeout_s):
        attempts.append(1)
        raise HTTPError(url, status, "provider error", {}, io.BytesIO(b"provider error"))

    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    gen = XAIImageGen(
        requester=request,
        downloader=_write_test_png,
        max_retries=3,
        retry_backoff_s=0.0,
        use_files_api=False,
    )
    with pytest.raises(RuntimeError, match=f"HTTP {status}"):
        gen.generate("terminal HTTP", out_path=str(tmp_path / "out.png"))

    assert len(attempts) == 1
