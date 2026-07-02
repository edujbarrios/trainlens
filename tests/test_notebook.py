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


def test_build_llm_report_uses_local_report_as_context(monkeypatch):
    monkeypatch.delenv("TRAINLENS_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("TRAINLENS_LLM_API_KEY", raising=False)

    report = build_llm_report(
        {
            "history": {"train_loss": [0.62, 0.18], "eval_loss": [0.48, 0.57]},
            "dataset_name": "ag_news",
        }
    )

    assert "TrainLens Report" in report.markdown
    assert "LLM explanation skipped" in report.markdown
    assert report.result.metrics["validation_loss"] == 0.57
