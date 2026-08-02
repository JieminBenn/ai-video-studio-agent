"""Wiring tests for the native reference-conditioned Gemini image provider.

The live generate_content call is paid + non-deterministic, so it is never unit-tested
(verified manually with GEMINI_API_KEY). Here we only assert selection, lazy
construction, the configured model, and the offline reference-count guard.
"""

from types import SimpleNamespace

import pytest

from studio_agent.cli import build_providers
from studio_agent.providers.gemini import GeminiImageGen, DEFAULT_MODEL, FALLBACK_MODEL


def _fake_response(*, image=None, text=None, finish_reason="STOP"):
    """A duck-typed stand-in for a google-genai generate_content response."""
    part = SimpleNamespace(
        inline_data=SimpleNamespace(data=image) if image else None,
        text=text,
    )
    candidate = SimpleNamespace(
        content=SimpleNamespace(parts=[part]),
        finish_reason=finish_reason,
        safety_ratings=None,
    )
    return SimpleNamespace(candidates=[candidate], prompt_feedback=None)


def test_build_providers_selects_native_gemini_image():
    providers = build_providers({
        "llm": "fake",
        "image": "gemini",
        "image_model": "gemini-3.1-flash-image",
    })
    assert isinstance(providers.image, GeminiImageGen)
    assert providers.image.model == "gemini-3.1-flash-image"


def test_gemini_defaults_and_fallback_model_names():
    assert DEFAULT_MODEL == "gemini-3.1-flash-image"
    assert FALLBACK_MODEL == "gemini-2.5-flash-image"


def test_gemini_constructs_without_network_or_key():
    gen = GeminiImageGen()
    assert gen.model == DEFAULT_MODEL
    assert gen.name == "gemini"


def test_gemini_guards_too_many_references_offline(tmp_path):
    gen = GeminiImageGen(max_reference_images=1)
    with pytest.raises(ValueError) as exc:
        gen.generate("p", out_path=str(tmp_path / "o.png"),
                     reference_images=["a.png", "b.png"])
    msg = str(exc.value)
    assert gen.model in msg
    assert "1" in msg and "2" in msg  # limit and count


def test_gemini_retries_a_transient_no_image_response(tmp_path):
    # Gemini image models intermittently return a text-only (no image) response;
    # a single transient miss must not abort the whole paid run.
    png = b"\x89PNG\r\n\x1a\nfake-image-bytes"
    responses = [
        _fake_response(text="Here is a description instead of an image."),
        _fake_response(image=png),
    ]
    calls = {"n": 0}

    def responder(*, prompt, reference_images):
        resp = responses[calls["n"]]
        calls["n"] += 1
        return resp

    gen = GeminiImageGen(responder=responder, max_attempts=3)
    out = tmp_path / "o.png"
    result = gen.generate("a lantern room", out_path=str(out))

    assert calls["n"] == 2  # retried once
    assert out.read_bytes() == png
    assert result.content == str(out)


def test_gemini_raises_with_diagnostics_when_all_attempts_have_no_image(tmp_path):
    def responder(*, prompt, reference_images):
        return _fake_response(text="I cannot generate that.", finish_reason="SAFETY")

    gen = GeminiImageGen(responder=responder, max_attempts=2)
    with pytest.raises(RuntimeError) as exc:
        gen.generate("p", out_path=str(tmp_path / "o.png"))
    msg = str(exc.value)
    assert "2" in msg  # number of attempts surfaced
    assert "SAFETY" in msg  # finish reason surfaced for diagnosis
