from trainlens.notebook import build_live_report


def test_build_live_report_returns_markdown_and_dashboard():
    report = build_live_report(
        {
            "history": {"train_loss": [2.0, 1.6], "eval_loss": [2.1, 1.8]},
            "training_trace": [{"step": 1, "loss": 2.0}, {"step": 2, "eval_loss": 1.8}],
        }
    )

    assert "TrainLens Report" in report.markdown
    assert "trainlens-dark-dashboard" in report.dashboard_html
    assert report.result.metrics["train_loss"] == 1.6
