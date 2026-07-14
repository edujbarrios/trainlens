from trainlens.notebook import build_improvement_ideas, build_llm_report, build_paper_report


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

    def fake_explain(
        markdown: str,
        *,
        mode: str = "paper_report",
        require_provider: bool = False,
    ) -> str:
        captured["markdown"] = markdown
        captured["mode"] = mode
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
    assert captured["mode"] == "paper_report"
    assert captured["require_provider"] is True
    assert "TrainLens Notebook Context" in str(captured["markdown"])
    assert "Contrastive loss is regressing" not in str(captured["markdown"])


def test_build_paper_report_uses_paper_mode(monkeypatch):
    captured: dict[str, object] = {}

    def fake_explain(
        markdown: str,
        *,
        mode: str = "paper_report",
        require_provider: bool = False,
    ) -> str:
        captured["mode"] = mode
        captured["require_provider"] = require_provider
        return "## TrainLens Scientific Report"

    monkeypatch.setattr("trainlens.notebook.explain_with_llm", fake_explain)

    report = build_paper_report({"history": {"train_loss": [1.0], "eval_loss": [1.2]}})

    assert report.markdown == "## TrainLens Scientific Report"
    assert captured["mode"] == "paper_report"
    assert captured["require_provider"] is True


def test_build_improvement_ideas_uses_improvement_mode(monkeypatch):
    captured: dict[str, object] = {}

    def fake_explain(
        markdown: str,
        *,
        mode: str = "paper_report",
        require_provider: bool = False,
    ) -> str:
        captured["mode"] = mode
        captured["require_provider"] = require_provider
        return "## TrainLens Improvement Ideas"

    monkeypatch.setattr("trainlens.notebook.explain_with_llm", fake_explain)

    report = build_improvement_ideas({"history": {"train_loss": [1.0], "eval_loss": [1.2]}})

    assert report.markdown == "## TrainLens Improvement Ideas"
    assert captured["mode"] == "improvement_ideas"
    assert captured["require_provider"] is True


def test_build_paper_report_forwards_metric_point_budget(monkeypatch):
    captured: dict[str, object] = {}

    def fake_explain(
        markdown: str,
        *,
        mode: str = "paper_report",
        require_provider: bool = False,
    ) -> str:
        captured["markdown"] = markdown
        return "## TrainLens Scientific Report"

    monkeypatch.setattr("trainlens.notebook.explain_with_llm", fake_explain)

    build_paper_report(
        {"history": {"loss": [1.0, 0.8, 0.6, 0.4]}},
        max_metric_points=2,
    )

    assert "ordered_sample=[1, 0.4]" in str(captured["markdown"])
