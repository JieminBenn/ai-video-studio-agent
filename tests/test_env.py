"""Tests for local .env loading without leaking or overriding secrets."""

import os

from studio_agent.env import load_dotenv, parse_dotenv


def test_parse_dotenv_accepts_common_env_syntax():
    parsed = parse_dotenv(
        """
        # comment
        OPENAI_API_KEY=sk-test
        export GEMINI_API_KEY="gemini-test"
        CUSTOM_PROVIDER_KEY='custom-test'
        EMPTY_KEY=
        """
    )

    assert parsed == {
        "OPENAI_API_KEY": "sk-test",
        "GEMINI_API_KEY": "gemini-test",
        "CUSTOM_PROVIDER_KEY": "custom-test",
        "EMPTY_KEY": "",
    }


def test_load_dotenv_loads_any_valid_key_without_overriding_shell_env(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "OPENAI_API_KEY=from-file\n"
        "CUSTOM_TEST_PROVIDER_KEY=custom-secret\n"
    )
    monkeypatch.setenv("OPENAI_API_KEY", "from-shell")
    monkeypatch.delenv("CUSTOM_TEST_PROVIDER_KEY", raising=False)

    loaded = load_dotenv(env_path)

    assert os.environ["OPENAI_API_KEY"] == "from-shell"
    assert os.environ["CUSTOM_TEST_PROVIDER_KEY"] == "custom-secret"
    assert loaded == ["CUSTOM_TEST_PROVIDER_KEY"]
