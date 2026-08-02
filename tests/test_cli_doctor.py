"""Tests for secret-safe CLI diagnostics."""

from studio_agent import cli
from studio_agent import env as env_support


def test_doctor_reports_key_presence_without_printing_values(tmp_path, monkeypatch, capsys):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "OPENAI_API_KEY=sk-secret-value\n"
        "CUSTOM_PROVIDER_KEY=custom-secret-value\n"
    )
    monkeypatch.setattr(env_support, "DEFAULT_ENV_PATH", env_path)
    for key in env_support.KNOWN_API_KEYS:
        monkeypatch.delenv(key, raising=False)

    code = cli.main(["doctor"])

    out = capsys.readouterr().out
    assert code == 0
    assert "OPENAI_API_KEY: set" in out
    assert "DEEPSEEK_API_KEY: missing" in out
    assert "GEMINI_API_KEY: missing" in out
    assert "XAI_API_KEY: missing" in out
    assert "DOUBAO_API_KEY: missing" in out
    assert "ARK_API_KEY: missing" in out
    assert "BYTEPLUS_ARK_API_KEY: missing" in out
    assert "VLLM_API_KEY: missing" in out
    assert "sk-secret-value" not in out
    assert "custom-secret-value" not in out
