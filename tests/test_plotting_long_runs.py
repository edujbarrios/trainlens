from __future__ import annotations

import pytest

from trainlens import plot_training_curves, training_curves

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


def test_auto_axis_is_observation_when_some_steps_unknown() -> None:
    namespace = {
        "log_history": [
            {"step": 100, "loss": 1.0, "eval_loss": 1.2},
            {"step": "bad", "loss": 0.7, "eval_loss": 0.9},
            {"step": 300, "loss": 0.5, "eval_loss": 0.6},
        ],
    }
    curves = training_curves(namespace, metrics=("loss",))
    assert curves[0].steps == (100, None, 300)

    figure = plot_training_curves(namespace, metrics=("loss",))
    assert tuple(figure.axes[0].lines[0].get_xdata()) == (1, 2, 3)
    assert figure.axes[0].get_xlabel() == "Observation"


def test_explicit_step_axis_preserves_unknown_positions() -> None:
    namespace = {
        "log_history": [
            {"step": 100, "loss": 1.0, "eval_loss": 1.1},
            {"step": "bad", "loss": 0.9, "eval_loss": 1.0},
            {"step": 400, "loss": 0.6, "eval_loss": 0.65},
        ]
    }
    figure = plot_training_curves(namespace, metrics=("loss",), x_axis="step")
    axis = figure.axes[0]
    assert tuple(axis.lines[0].get_xdata()) == (100, None, 400)
    assert axis.lines[0].get_linestyle() == "None"
    assert axis.get_xlabel() == "Recorded step (unknown steps omitted)"


def test_plot_does_not_infer_steps_for_unstepped_curves() -> None:
    with pytest.raises(ValueError, match="requires recorded steps"):
        plot_training_curves(
            {"history": {"loss": [1.0, 0.8, 0.6]}},
            metrics=("loss",),
            x_axis="step",
        )


def test_plot_long_history_preserves_extrema_but_reduces_drawn_points() -> None:
    values = [1.0 - index * 0.0001 for index in range(5000)]
    values[222] = 10.0
    values[3411] = -8.0
    data = {"history": {"train_loss": values, "val_loss": values}}

    figure = plot_training_curves(data, metrics=("loss",), max_plot_points=200)
    train = figure.axes[0].lines[0]
    plotted = tuple(train.get_ydata())
    assert len(plotted) <= 200
    assert 10.0 in plotted
    assert -8.0 in plotted
    assert len(values) == 5000
    assert train.get_marker() == ""
    assert figure.axes[0].get_xlabel() == "Observation"


def test_plot_highlights_validation_best_not_test_metric() -> None:
    data = {
        "history": {
            "train_loss": [1.0, 0.6, 0.4],
            "val_loss": [1.2, 0.4, 0.7],
        },
        "test_metrics": {"loss": 0.2},
    }
    figure = plot_training_curves(data, metrics=("loss",))
    axis = figure.axes[0]
    assert [line.get_label() for line in axis.lines] == [
        "train", "validation", "test (held-out)"
    ]
    assert tuple(axis.lines[-1].get_xdata()) == (3,)
    assert len(axis.collections) == 1
    assert tuple(axis.collections[0].get_offsets()[0]) == (2, 0.4)


@pytest.mark.parametrize("invalid", [True, 3.5, "100"])
def test_plot_rejects_non_integer_budgets(invalid: object) -> None:
    with pytest.raises(TypeError, match="max_plot_points"):
        plot_training_curves(
            {"history": {"loss": [1.0, 0.5]}},
            max_plot_points=invalid,  # type: ignore[arg-type]
        )


def test_plot_rejects_too_small_budget_and_bad_axis_name() -> None:
    data = {"history": {"loss": [1.0, 0.5]}}
    with pytest.raises(ValueError, match="at least 4"):
        plot_training_curves(data, max_plot_points=3)
    with pytest.raises(ValueError, match="x_axis"):
        plot_training_curves(data, x_axis="bad")  # type: ignore[arg-type]
