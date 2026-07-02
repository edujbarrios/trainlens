from trainlens.notebook import build_live_report, build_llm_report


def test_build_live_report_returns_markdown():
    report = build_live_report(
        {
            "history": {"train_loss": [2.0, 1.6], "eval_loss": [2.1, 1.8]},
            "training_trace": [{"step": 1, "loss": 2.0}, {"step": 2, "eval_loss": 1.8}],
        }
    )

    assert "TrainLens Report" in report.markdown
    assert report.result.metrics["train_loss"] == 1.6


def test_build_live_report_respects_explicit_empty_namespace(monkeypatch):
    monkeypatch.setattr("trainlens.notebook.get_ipython", lambda: None)

    report = build_live_report({})

    assert "TrainLens Report" in report.markdown
    assert report.result.metrics == {}


def test_build_llm_report_uses_local_report_as_context(monkeypatch):
    monkeypatch.delenv("TRAINLENS_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("TRAINLENS_LLM_API_KEY", raising=False)

    try:
        build_llm_report(
            {
                "history": {"train_loss": [0.62, 0.18], "eval_loss": [0.48, 0.57]},
                "dataset_name": "ag_news",
            }
        )
    except RuntimeError as exc:
        assert "TRAINLENS_LLM_API_KEY" in str(exc)
    else:
        raise AssertionError("Expected missing provider config to raise")


def test_build_llm_report_sends_context_to_llm(monkeypatch):
    captured: dict[str, object] = {}

    def fake_explain(markdown: str, *, require_provider: bool = False) -> str:
        captured["markdown"] = markdown
        captured["require_provider"] = require_provider
        return "## TrainLens Report\n\nLLM-only report."

    monkeypatch.setattr("trainlens.notebook.explain_with_llm", fake_explain)

    report = build_llm_report(
        {
            "history": {"train_loss": [0.62, 0.18], "eval_loss": [0.48, 0.57]},
            "dataset_name": "ag_news",
            "model_name": "distilbert-base-uncased",
        }
    )

    assert report.markdown == "## TrainLens Report\n\nLLM-only report."
    assert report.result.metrics["validation_loss"] == 0.57
    assert captured["require_provider"] is True
    assert "TrainLens Notebook Context" in str(captured["markdown"])
    assert "Contrastive loss is regressing" not in str(captured["markdown"])
