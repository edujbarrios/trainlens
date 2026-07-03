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

    assert LLMConfig.from_env() is None


def test_llm_config_defaults_blank_model_to_auto(monkeypatch):
    monkeypatch.setenv("TRAINLENS_LLM_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("TRAINLENS_LLM_API_KEY", "secret-key")
    monkeypatch.setenv("TRAINLENS_LLM_MODEL", "   ")

    config = LLMConfig.from_env()

    assert config is not None
    assert config.model == "auto"


def test_llm_config_defaults_invalid_timeout(monkeypatch):
    monkeypatch.setenv("TRAINLENS_LLM_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("TRAINLENS_LLM_API_KEY", "secret-key")
    monkeypatch.setenv("TRAINLENS_LLM_TIMEOUT_SECONDS", "-1")

    config = LLMConfig.from_env()

    assert config is not None
    assert config.timeout_seconds == 120.0
