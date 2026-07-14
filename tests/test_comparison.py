from __future__ import annotations

import json

import pytest

from trainlens import compare_runs, render_report, render_run_comparison
from trainlens.models.analysis import AnalysisResult
from trainlens.models.metric import MetricSeries
from trainlens.models.run import TrainingRun


def test_compare_runs_detects_improvement_and_regression() -> None:
    baseline = AnalysisResult(
        model_name="baseline",
        metrics={"validation_loss": 0.5, "validation_accuracy": 0.82},
    )
    experiment = AnalysisResult(
        model_name="experiment",
        metrics={"validation_loss": 0.42, "validation_accuracy": 0.8},
    )

    comparison = compare_runs(baseline, experiment)

    by_name = {item.name: item for item in comparison.metrics}
    assert by_name["validation_loss"].direction == "improved"
    assert by_name["validation_loss"].magnitude == "material"
    assert by_name["validation_accuracy"].direction == "regressed"
    assert comparison.improvements == (by_name["validation_loss"],)
    assert comparison.regressions == (by_name["validation_accuracy"],)
    assert "Material improvement detected in validation_loss." in comparison.summary


def test_compare_runs_marks_new_and_removed_metrics() -> None:
    comparison = compare_runs(
        {"loss": 1.0, "accuracy": 0.7, "train_runtime": 12.0},
        {"loss": 0.9, "f1": 0.66, "train_runtime": 10.0},
    )

    by_name = {item.name: item for item in comparison.metrics}
    assert by_name["accuracy"].direction == "removed"
    assert by_name["f1"].direction == "new"
    assert by_name["train_runtime"].direction == "unknown"
    assert comparison.notes


def test_compare_runs_does_not_classify_partial_metric_name_matches() -> None:
    comparison = compare_runs(
        {"lossless_compression": 0.5, "maple_syrup": 0.5},
        {"lossless_compression": 0.4, "maple_syrup": 0.6},
    )

    assert {item.direction for item in comparison.metrics} == {"unknown"}


def test_compare_runs_accepts_training_run_metric_series() -> None:
    baseline = TrainingRun(
        model_name="baseline",
        metrics=(MetricSeries("validation_loss", (0.7, 0.6)),),
    )
    experiment = TrainingRun(
        model_name="experiment",
        metrics=(MetricSeries("validation_loss", (0.7, 0.52)),),
    )

    comparison = compare_runs(baseline, experiment)

    assert comparison.baseline_name == "baseline"
    assert comparison.experiment_name == "experiment"
    assert comparison.metrics[0].delta == pytest.approx(-0.08)


def test_render_run_comparison_outputs_markdown_table() -> None:
    comparison = compare_runs(
        {"validation_loss": 0.5},
        {"validation_loss": 0.4},
        baseline_name="before",
        experiment_name="after",
    )

    markdown = render_run_comparison(comparison)

    assert "## TrainLens Run Comparison" in markdown
    assert "**Baseline:** before" in markdown
    assert "| validation_loss | 0.5 | 0.4 | -0.1 | -20.0% | improved | material |" in markdown


def test_render_report_exports_run_comparison_json_and_html() -> None:
    comparison = compare_runs({"loss": 1.0}, {"loss": 0.8})

    payload = json.loads(str(render_report(comparison, format="json")))
    html = str(render_report(comparison, format="html"))

    assert payload["metrics"][0]["name"] == "loss"
    assert payload["improvements"][0]["direction"] == "improved"
    assert "<h2>TrainLens Run Comparison</h2>" in html
