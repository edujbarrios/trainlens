from trainlens.llm.enhancer import explain_with_llm


def test_explain_with_llm_skips_when_provider_config_is_missing(monkeypatch):
    monkeypatch.delenv("TRAINLENS_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("TRAINLENS_LLM_API_KEY", raising=False)
    monkeypatch.delenv("TRAINLENS_LLM_MODEL", raising=False)

    report = "## TrainLens Report\n\n- eval_loss=0.57"

    explained = explain_with_llm(report)

    assert report in explained
    assert "LLM explanation skipped" in explained


def test_explain_with_llm_can_require_provider_config(monkeypatch):
    monkeypatch.delenv("TRAINLENS_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("TRAINLENS_LLM_API_KEY", raising=False)

    try:
        explain_with_llm("report", require_provider=True)
    except RuntimeError as exc:
        assert "TRAINLENS_LLM_API_KEY" in str(exc)
    else:
        raise AssertionError("Expected missing provider config to raise")


def test_explain_with_llm_strict_mode_raises_on_provider_error(monkeypatch):
    class FailingProvider:
        def __init__(self, _config: object) -> None:
            pass

        def explain(self, _markdown_report: str, *, mode: str = "paper_report") -> str:
            raise ValueError("provider unavailable")

    monkeypatch.setenv("TRAINLENS_LLM_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("TRAINLENS_LLM_API_KEY", "secret-key")
    monkeypatch.setenv("TRAINLENS_LLM_MODEL", "trainlens-test-model")
    monkeypatch.setattr("trainlens.llm.enhancer.OpenAICompatibleProvider", FailingProvider)

    try:
        explain_with_llm("report", require_provider=True)
    except RuntimeError as exc:
        assert "provider unavailable" in str(exc)
    else:
        raise AssertionError("Expected strict provider errors to raise")
