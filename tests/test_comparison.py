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


def test_compare_runs_reports_material_unknown_metric_movement() -> None:
    comparison = compare_runs(
        {"train_runtime": 10.0},
        {"train_runtime": 12.0},
    )

    assert comparison.metrics[0].direction == "unknown"
    assert comparison.metrics[0].magnitude == "material"
    assert comparison.summary == (
        "Material change detected in train_runtime, but optimization direction is unknown.",
    )
    assert "No material metric movement detected." not in comparison.summary


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


def test_compare_runs_ignores_non_finite_metrics() -> None:
    comparison = compare_runs(
        {"loss": 1.0, "accuracy": float("nan")},
        {"loss": float("inf"), "accuracy": 0.8},
    )

    by_name = {item.name: item for item in comparison.metrics}
    assert by_name["loss"].direction == "removed"
    assert by_name["accuracy"].direction == "new"
    assert all(item.delta is None for item in comparison.metrics)
    assert "NaN" not in str(render_report(comparison, format="json"))
    assert "Infinity" not in str(render_report(comparison, format="json"))


def test_compare_runs_ignores_boolean_and_non_numeric_metrics() -> None:
    comparison = compare_runs(
        {"accuracy": False, "loss": "not-a-number"},
        {"accuracy": True, "loss": None},
    )

    assert comparison.metrics == ()


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


def test_run_comparison_renders_itself_as_markdown_in_notebooks() -> None:
    comparison = compare_runs(
        {"loss": 0.52, "accuracy": 0.84},
        {"loss": 0.41, "accuracy": 0.89},
        baseline_name="baseline",
        experiment_name="new run",
    )

    markdown = comparison.to_markdown()

    assert markdown == render_run_comparison(comparison)
    assert comparison._repr_markdown_() == markdown
    assert "**Experiment:** new run" in markdown
    assert "| accuracy | 0.84 | 0.89 | +0.05 | +6.0% | improved | material |" in markdown


def test_render_report_exports_run_comparison_json_and_html() -> None:
    comparison = compare_runs({"loss": 1.0}, {"loss": 0.8})

    payload = json.loads(str(render_report(comparison, format="json")))
    html = str(render_report(comparison, format="html"))

    assert payload["metrics"][0]["name"] == "loss"
    assert payload["improvements"][0]["direction"] == "improved"
    assert "<h2>TrainLens Run Comparison</h2>" in html


@pytest.mark.parametrize("metric", ["mae", "mape", "mse", "msle", "rmse"])
def test_compare_runs_treats_common_error_metrics_as_lower_is_better(metric) -> None:
    comparison = compare_runs({metric: 1.0}, {metric: 0.8})

    assert comparison.metrics[0].direction == "improved"


def test_run_comparison_escapes_metric_names_containing_pipes() -> None:
    comparison = compare_runs({"precision|recall": 0.5}, {"precision|recall": 0.6})

    markdown = render_run_comparison(comparison)
    html = str(render_report(comparison, format="html"))

    assert r"precision\|recall" in markdown
    assert html.count("<td>") == 7
    assert "precision|recall" in html
