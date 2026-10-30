import pytest

from trainlens.llm.config import LLMConfig


def test_llm_config_strips_environment_values(monkeypatch):
    monkeypatch.setenv("TRAINLENS_LLM_BASE_URL", " https://api.example.com/v1/ ")
    monkeypatch.setenv("TRAINLENS_LLM_API_KEY", " secret-key ")
    monkeypatch.setenv("TRAINLENS_LLM_MODEL", " trainlens-test-model ")
    monkeypatch.setenv("TRAINLENS_LLM_TIMEOUT_SECONDS", "180")

    config = LLMConfig.from_env()

    assert config == LLMConfig(
        base_url="https://api.example.com/v1",
        api_key="secret-key",
        model="trainlens-test-model",
        timeout_seconds=180.0,
    )


def test_llm_config_ignores_blank_required_values(monkeypatch):
    monkeypatch.setenv("TRAINLENS_LLM_BASE_URL", "   ")
    monkeypatch.setenv("TRAINLENS_LLM_API_KEY", "secret-key")
    monkeypatch.setenv("TRAINLENS_LLM_MODEL", "trainlens-test-model")

    assert LLMConfig.from_env() is None


def test_llm_config_requires_model(monkeypatch):
    monkeypatch.setenv("TRAINLENS_LLM_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("TRAINLENS_LLM_API_KEY", "secret-key")
    monkeypatch.setenv("TRAINLENS_LLM_MODEL", "   ")

    assert LLMConfig.from_env() is None


def test_llm_config_defaults_invalid_timeout(monkeypatch):
    monkeypatch.setenv("TRAINLENS_LLM_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("TRAINLENS_LLM_API_KEY", "secret-key")
    monkeypatch.setenv("TRAINLENS_LLM_MODEL", "trainlens-test-model")
    monkeypatch.setenv("TRAINLENS_LLM_TIMEOUT_SECONDS", "-1")

    config = LLMConfig.from_env()

    assert config is not None
    assert config.timeout_seconds == 120.0


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_llm_config_rejects_non_finite_timeouts(monkeypatch, value: str) -> None:
    monkeypatch.setenv("TRAINLENS_LLM_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("TRAINLENS_LLM_API_KEY", "secret-key")
    monkeypatch.setenv("TRAINLENS_LLM_MODEL", "trainlens-test-model")
    monkeypatch.setenv("TRAINLENS_LLM_TIMEOUT_SECONDS", value)

    config = LLMConfig.from_env()

    assert config is not None
    assert config.timeout_seconds == 120.0


def test_llm_config_reads_key_from_file(monkeypatch, tmp_path):
    secret = tmp_path / "api-key"
    secret.write_text("  file-secret\n", encoding="utf-8")
    monkeypatch.setenv("TRAINLENS_LLM_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("TRAINLENS_LLM_MODEL", "remote-model")
    monkeypatch.delenv("TRAINLENS_LLM_API_KEY", raising=False)
    monkeypatch.setenv("TRAINLENS_LLM_API_KEY_FILE", str(secret))

    config = LLMConfig.from_env()

    assert config is not None
    assert config.api_key == "file-secret"


def test_llm_config_environment_key_overrides_file(monkeypatch, tmp_path):
    secret = tmp_path / "api-key"
    secret.write_text("file-secret", encoding="utf-8")
    monkeypatch.setenv("TRAINLENS_LLM_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("TRAINLENS_LLM_MODEL", "remote-model")
    monkeypatch.setenv("TRAINLENS_LLM_API_KEY", "env-secret")
    monkeypatch.setenv("TRAINLENS_LLM_API_KEY_FILE", str(secret))

    config = LLMConfig.from_env()

    assert config is not None
    assert config.api_key == "env-secret"


@pytest.mark.parametrize("value", ["  ", "line-one\nline-two", "x" * 4097])
def test_llm_config_rejects_invalid_key_files(monkeypatch, tmp_path, value):
    secret = tmp_path / "api-key"
    secret.write_text(value, encoding="utf-8")
    monkeypatch.setenv("TRAINLENS_LLM_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("TRAINLENS_LLM_MODEL", "remote-model")
    monkeypatch.delenv("TRAINLENS_LLM_API_KEY", raising=False)
    monkeypatch.setenv("TRAINLENS_LLM_API_KEY_FILE", str(secret))

    with pytest.raises(ValueError, match="TRAINLENS_LLM_API_KEY_FILE"):
        LLMConfig.from_env()


def test_llm_config_rejects_missing_key_file(monkeypatch, tmp_path):
    monkeypatch.setenv("TRAINLENS_LLM_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("TRAINLENS_LLM_MODEL", "remote-model")
    monkeypatch.delenv("TRAINLENS_LLM_API_KEY", raising=False)
    monkeypatch.setenv("TRAINLENS_LLM_API_KEY_FILE", str(tmp_path / "missing"))

    with pytest.raises(ValueError, match="TRAINLENS_LLM_API_KEY_FILE"):
        LLMConfig.from_env()


def test_llm_config_does_not_load_key_file_without_provider(monkeypatch, tmp_path):
    monkeypatch.delenv("TRAINLENS_LLM_BASE_URL", raising=False)
    monkeypatch.setenv("TRAINLENS_LLM_MODEL", "remote-model")
    monkeypatch.delenv("TRAINLENS_LLM_API_KEY", raising=False)
    monkeypatch.setenv("TRAINLENS_LLM_API_KEY_FILE", str(tmp_path / "missing"))

    assert LLMConfig.from_env() is None
