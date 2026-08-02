"""Tests for OpenAI-compatible LLM providers without network calls."""

import pytest

from studio_agent import cli
from studio_agent.providers.openai_compatible import (
    OpenAICompatibleLLM,
    _consume_sse_stream,
    parse_json_content,
)


def _response(content):
    return {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"total_tokens": 12},
    }


def test_complete_posts_chat_completion_shape():
    calls = []

    def transport(url, payload, headers, timeout_s):
        calls.append((url, payload, headers, timeout_s))
        return _response("hello")

    llm = OpenAICompatibleLLM(
        base_url="http://localhost:11434/v1",
        model="qwen-local",
        api_key="secret",
        transport=transport,
    )

    gen = llm.complete("write")

    assert gen.content == "hello"
    url, payload, headers, timeout_s = calls[0]
    assert url == "http://localhost:11434/v1/chat/completions"
    assert payload["model"] == "qwen-local"
    assert payload["messages"][-1] == {"role": "user", "content": "write"}
    assert headers["Authorization"] == "Bearer secret"
    assert timeout_s == 120.0


def test_deepseek_v4_pro_tracks_cached_cny_usage_in_usd():
    response = {
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 1_000_000,
            "completion_tokens": 500_000,
            "prompt_tokens_details": {"cached_tokens": 250_000},
        },
        "model": "deepseek-v4-pro-260425",
    }
    llm = OpenAICompatibleLLM(
        base_url="https://ark.example/v1",
        model="deepseek-v4-pro-260425",
        api_key="k",
        transport=lambda *args: response,
        cost_per_million_input=12.0,
        cost_per_million_cached_input=1.0,
        cost_per_million_output=24.0,
        cost_currency="CNY",
        usd_per_native_unit=0.14,
        pricing_source="official",
        pricing_as_of="2026-07-01",
    )

    gen = llm.complete("write")

    assert gen.cost_usd == 2.975
    assert gen.meta["cost_tracking"]["native_cost"] == 21.25
    assert gen.meta["cost_tracking"]["usage"]["cached_input_tokens"] == 250_000


def test_openai_compatible_llm_marks_missing_usage_incomplete():
    response = {
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "model": "priced-model",
    }
    llm = OpenAICompatibleLLM(
        base_url="https://example.test/v1",
        model="priced-model",
        api_key="k",
        transport=lambda *args: response,
        cost_per_million_input=1.0,
        cost_per_million_output=2.0,
    )

    gen = llm.complete("write")

    assert gen.cost_usd == 0.0
    assert gen.meta["cost_tracking"]["usage_missing"] is True


def test_configured_remote_provider_reports_missing_key_before_transport(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    called = False

    def transport(*args):
        nonlocal called
        called = True
        return _response("should not happen")

    llm = OpenAICompatibleLLM(
        base_url="https://api.x.ai/v1",
        model="grok-4.3",
        api_key_env="XAI_API_KEY",
        transport=transport,
    )

    with pytest.raises(RuntimeError, match="XAI_API_KEY"):
        llm.complete("write")
    assert called is False


def test_complete_json_parses_markdown_fenced_json():
    llm = OpenAICompatibleLLM(
        base_url="http://localhost:8000/v1",
        model="local",
        transport=lambda *args: _response('```json\n{"ok": true}\n```'),
    )

    gen = llm.complete_json("return json")

    assert gen.content == {"ok": True}
    assert gen.meta["json_attempts"] == 1


def test_complete_json_can_omit_response_format_for_unsupported_models():
    calls = []

    def transport(url, payload, headers, timeout_s):
        calls.append(payload)
        return _response('{"ok": true}')

    llm = OpenAICompatibleLLM(
        base_url="https://ark.ap-southeast.bytepluses.com/api/v3",
        model="deepseek-v4-flash-260425",
        json_response_format=False,
        transport=transport,
    )

    gen = llm.complete_json("return json")

    assert gen.content == {"ok": True}
    assert "response_format" not in calls[0]


def test_complete_json_retries_bad_json_before_succeeding():
    responses = iter([_response("not json"), _response('{"ok": true}')])
    prompts = []

    def transport(url, payload, headers, timeout_s):
        prompts.append(payload["messages"][-1]["content"])
        return next(responses)

    llm = OpenAICompatibleLLM(
        base_url="http://localhost:8000/v1",
        model="local",
        max_json_attempts=2,
        transport=transport,
    )

    gen = llm.complete_json("return json")

    assert gen.content == {"ok": True}
    assert gen.meta["json_attempts"] == 2
    assert "Previous JSON parse error" in prompts[1]
    assert "single-line JSON strings" in prompts[1]


def test_complete_json_raises_after_bad_json():
    llm = OpenAICompatibleLLM(
        base_url="http://localhost:8000/v1",
        model="local",
        max_json_attempts=2,
        transport=lambda *args: _response("not json"),
    )

    with pytest.raises(ValueError, match="did not return valid JSON"):
        llm.complete_json("return json")


def test_parse_json_content_extracts_first_object_from_prose():
    assert parse_json_content('Sure:\n{"a": 1}\nthanks') == {"a": 1}


def test_cli_builds_openai_compatible_llm_profile():
    providers = cli.build_providers({
        "llm": "openai-compatible",
        "llm_base_url": "http://localhost:11434/v1",
        "llm_model": "qwen-local",
        "image": "fake",
        "video": "fake",
        "tts": "fake",
        "music": "fake",
        "vlm": "fake",
    })

    assert isinstance(providers.llm, OpenAICompatibleLLM)
    assert providers.llm.base_url == "http://localhost:11434/v1"
    assert providers.llm.model == "qwen-local"


def test_cli_passes_json_response_format_flag_to_llm():
    providers = cli.build_providers({
        "llm": "openai-compatible",
        "llm_base_url": "https://ark.ap-southeast.bytepluses.com/api/v3",
        "llm_model": "deepseek-v4-flash-260425",
        "llm_json_response_format": False,
        "image": "fake",
        "video": "fake",
        "tts": "fake",
        "music": "fake",
        "vlm": "fake",
    })

    assert providers.llm.json_response_format is False


def test_cli_passes_native_currency_pricing_to_llm():
    providers = cli.build_providers({
        "llm": "openai-compatible",
        "llm_base_url": "https://ark.example/v1",
        "llm_model": "deepseek-v4-pro-260425",
        "llm_cost_per_million_input": 12.0,
        "llm_cost_per_million_cached_input": 1.0,
        "llm_cost_per_million_output": 24.0,
        "llm_cost_currency": "CNY",
        "llm_usd_per_native_unit": 0.14,
        "llm_pricing_source": "official",
        "llm_pricing_as_of": "2026-07-01",
        "image": "fake",
        "video": "fake",
        "tts": "fake",
        "music": "fake",
        "vlm": "fake",
    })

    assert providers.llm.cost_per_million_input == 12.0
    assert providers.llm.cost_per_million_cached_input == 1.0
    assert providers.llm.cost_per_million_output == 24.0
    assert providers.llm.cost_currency == "CNY"
    assert providers.llm.usd_per_native_unit == 0.14


def test_complete_retries_transient_connection_blip_then_succeeds():
    # A DNS/dropped-connection hiccup (URLError wrapping gaierror/ConnectionError) is transient
    # and must be retried with backoff, not abort the stage — mirroring the video Ark path.
    import socket
    import urllib.error

    attempts = []

    def transport(url, payload, headers, timeout_s):
        attempts.append(1)
        if len(attempts) == 1:
            raise urllib.error.URLError(socket.gaierror("temporary name resolution failure"))
        return _response("recovered")

    llm = OpenAICompatibleLLM(
        base_url="https://ark.example/v1", model="deepseek", api_key="k",
        transport=transport, max_retries=3, retry_backoff_s=0.0,
    )

    gen = llm.complete("hi")
    assert gen.content == "recovered"
    assert len(attempts) == 2


def test_complete_retries_connection_reset_then_succeeds():
    # http.client.RemoteDisconnected ("Remote end closed connection without response") is a
    # ConnectionResetError, NOT a urllib URLError — the old retry classifier missed it, so a
    # transient server-side connection drop aborted the whole stage with the raw message.
    attempts = []

    def transport(url, payload, headers, timeout_s):
        attempts.append(1)
        if len(attempts) == 1:
            raise ConnectionResetError("Remote end closed connection without response")
        return _response("recovered")

    llm = OpenAICompatibleLLM(
        base_url="https://ark.example/v1", model="deepseek", api_key="k",
        transport=transport, max_retries=3, retry_backoff_s=0.0,
    )

    gen = llm.complete("hi")
    assert gen.content == "recovered"
    assert len(attempts) == 2


def test_read_timeout_raises_a_clear_terminal_error_without_retrying():
    # A read-phase socket timeout ("The read operation timed out") means the model itself was
    # too slow; retrying re-sends the whole prompt and will time out again, so it is terminal —
    # but the raw socket message must be replaced with an actionable one naming the timeout.
    attempts = []

    def transport(url, payload, headers, timeout_s):
        attempts.append(1)
        raise TimeoutError("The read operation timed out")

    llm = OpenAICompatibleLLM(
        base_url="https://ark.example/v1", model="deepseek-v4-pro", api_key="k",
        transport=transport, max_retries=3, retry_backoff_s=0.0, timeout_s=180.0,
    )

    with pytest.raises(RuntimeError, match="timed out after 180"):
        llm.complete("hi")
    assert len(attempts) == 1


def test_consume_sse_stream_assembles_content_finish_and_usage():
    # A reasoning model streams reasoning_content first (which keeps the socket alive but is not
    # part of the answer), then the real content deltas, a finish_reason, and a final usage-only
    # chunk (from stream_options.include_usage), terminated by [DONE].
    lines = [
        b'data: {"choices":[{"delta":{"reasoning_content":"thinking..."}}]}\n',
        b'data: {"choices":[{"delta":{"content":"He"},"finish_reason":null}]}\n',
        b'\n',  # SSE keep-alive / blank separator line
        b'data: {"choices":[{"delta":{"content":"llo"},"finish_reason":"stop"}]}\n',
        b'data: {"choices":[],"usage":{"total_tokens":7},"model":"deepseek-v4-pro-260425"}\n',
        b'data: [DONE]\n',
    ]

    data = _consume_sse_stream(lines, model="deepseek-v4-pro-260425")

    assert data["choices"][0]["message"]["content"] == "Hello"
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["usage"] == {"total_tokens": 7}
    assert data["model"] == "deepseek-v4-pro-260425"


def test_streaming_llm_sets_stream_payload_and_parses_content():
    calls = []

    def transport(url, payload, headers, timeout_s):
        calls.append(payload)
        # Emulate what _urlopen_stream_transport returns after consuming the SSE body.
        return {
            "choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 3},
        }

    llm = OpenAICompatibleLLM(
        base_url="https://ark.example/v1",
        model="deepseek-v4-pro-260425",
        api_key="k",
        stream=True,
        transport=transport,
    )

    gen = llm.complete_json("return json")

    assert gen.content == {"ok": True}
    assert calls[0]["stream"] is True
    assert calls[0]["stream_options"] == {"include_usage": True}


def test_non_streaming_llm_omits_stream_payload():
    calls = []

    def transport(url, payload, headers, timeout_s):
        calls.append(payload)
        return _response("hi")

    llm = OpenAICompatibleLLM(
        base_url="https://ark.example/v1",
        model="deepseek",
        api_key="k",
        stream=False,
        transport=transport,
    )

    llm.complete("hi")

    assert "stream" not in calls[0]
    assert "stream_options" not in calls[0]


def test_cli_streams_by_default_and_honors_opt_out():
    streaming = cli.build_providers({
        "llm": "openai-compatible",
        "llm_base_url": "https://ark.example/v1",
        "llm_model": "deepseek-v4-pro-260425",
        "image": "fake", "video": "fake", "tts": "fake", "music": "fake", "vlm": "fake",
    })
    assert streaming.llm.stream is True

    opted_out = cli.build_providers({
        "llm": "openai-compatible",
        "llm_base_url": "https://ark.example/v1",
        "llm_model": "deepseek-v4-pro-260425",
        "llm_stream": False,
        "image": "fake", "video": "fake", "tts": "fake", "music": "fake", "vlm": "fake",
    })
    assert opted_out.llm.stream is False


def test_http_4xx_stays_terminal_without_retry():
    import urllib.error

    attempts = []

    def transport(url, payload, headers, timeout_s):
        attempts.append(1)
        raise urllib.error.HTTPError(url, 400, "Bad Request", {}, None)

    llm = OpenAICompatibleLLM(
        base_url="https://ark.example/v1", model="deepseek", api_key="k",
        transport=transport, max_retries=3, retry_backoff_s=0.0,
    )

    with pytest.raises(RuntimeError, match="HTTP 400"):
        llm.complete("hi")
    assert len(attempts) == 1
