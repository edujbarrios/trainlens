from __future__ import annotations

import pytest

from trainlens import ExperimentRun, compare_runs, suggest_next_experiment
from trainlens.metric_semantics import metric_bounds, metric_direction


@pytest.mark.parametrize(
    "metric",
    [
        "loss",
        "error",
        "perplexity",
        "wer",
        "cer",
        "latency",
        "fad",
        "frechet_audio_distance",
        "mae",
        "mape",
        "mse",
        "msle",
        "rmse",
    ],
)
def test_lower_is_better_metrics_are_consistent_across_modules(metric: str) -> None:
    comparison = compare_runs({metric: 1.0}, {metric: 0.8})
    recommendation = suggest_next_experiment(
        [ExperimentRun(name="run", metrics={metric: 1.0})],
        objective_metric=metric,
    )

    assert metric_direction(metric) == "lower"
    assert comparison.metrics[0].direction == "improved"
    assert recommendation.success_criteria[0].operator == "<="


@pytest.mark.parametrize(
    "metric",
    ["accuracy", "acc", "auc", "f1", "precision", "recall", "score", "map", "ndcg"],
)
def test_higher_is_better_metrics_are_consistent_across_modules(metric: str) -> None:
    comparison = compare_runs({metric: 0.5}, {metric: 0.6})
    recommendation = suggest_next_experiment(
        [ExperimentRun(name="run", metrics={metric: 0.5})],
        objective_metric=metric,
    )

    assert metric_direction(metric) == "higher"
    assert comparison.metrics[0].direction == "improved"
    assert recommendation.success_criteria[0].operator == ">="


def test_shared_semantics_exposes_natural_bounds_for_bounded_metrics() -> None:
    assert metric_bounds("validation_accuracy") == (0.0, 1.0)
    assert metric_bounds("ndcg_at_10") == (0.0, 1.0)
    assert metric_bounds("validation_loss") == (0.0, None)
    assert metric_bounds("custom_metric") == (None, None)
