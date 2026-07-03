from trainlens.notebook import build_llm_report


def test_build_llm_report_requires_provider(monkeypatch):
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
