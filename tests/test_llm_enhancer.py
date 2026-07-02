from trainlens.llm.enhancer import explain_with_llm, maybe_enhance


def test_explain_with_llm_skips_when_provider_config_is_missing(monkeypatch):
    monkeypatch.delenv("TRAINLENS_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("TRAINLENS_LLM_API_KEY", raising=False)
    monkeypatch.delenv("TRAINLENS_LLM_MODEL", raising=False)

    report = "## TrainLens Report\n\n- eval_loss=0.57"

    explained = explain_with_llm(report)

    assert report in explained
    assert "LLM explanation skipped" in explained


def test_maybe_enhance_remains_backward_compatible(monkeypatch):
    monkeypatch.delenv("TRAINLENS_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("TRAINLENS_LLM_API_KEY", raising=False)

    assert maybe_enhance("report") == explain_with_llm("report")
