from __future__ import annotations

import pytest

from trainlens import TrainingRun, select_checkpoint
from trainlens.models.metric import MetricSeries


def test_select_checkpoint_finds_validation_optimum_and_late_degradation() -> None:
    series = MetricSeries(
        "validation_loss",
        (0.8, 0.6, 0.5, 0.52, 0.58),
        split="validation",
        steps=(100, 200, 300, 400, 500),
    )

    selection = select_checkpoint(series)

    assert selection.best_index == 2
    assert selection.best_step == 300
    assert selection.best_value == 0.5
    assert selection.overfit_after == 400
    assert selection.degradation_from_best == pytest.approx(0.08)
    assert selection.suggested_patience >= 1


def test_select_checkpoint_supports_training_run_and_higher_is_better() -> None:
    run = TrainingRun(
        run_id="accuracy-run",
        metrics=(
            MetricSeries(
                "validation_accuracy",
                (0.70, 0.78, 0.82, 0.81),
                split="validation",
                steps=(1, 2, 3, 4),
            ),
        ),
    )

    selection = select_checkpoint(run, metric="validation_accuracy")

    assert selection.direction == "higher"
    assert selection.best_step == 3
    assert selection.best_value == 0.82
    assert selection.overfit_after == 4


def test_select_checkpoint_supports_mapping_and_custom_direction() -> None:
    selection = select_checkpoint(
        {"custom_score": [1.0, 1.5, 1.4]},
        metric="custom_score",
        direction="max",
        min_delta=0.05,
    )

    assert selection.best_step == 1
    assert selection.best_value == 1.5


def test_select_checkpoint_rejects_test_selection_and_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="not test metrics"):
        select_checkpoint(
            MetricSeries("test_loss", (0.5, 0.4), split="test"),
            metric="test_loss",
        )
    with pytest.raises(ValueError, match="cannot infer"):
        select_checkpoint({"custom": [1.0, 2.0]}, metric="custom")
    with pytest.raises(ValueError, match="does not contain"):
        select_checkpoint(TrainingRun(run_id="empty"), metric="validation_loss")
    with pytest.raises(ValueError, match="finite and non-negative"):
        select_checkpoint({"validation_loss": [1.0]}, min_delta=-1.0)
    with pytest.raises(TypeError, match="numeric"):
        select_checkpoint({"validation_loss": [1.0]}, min_delta=True)
