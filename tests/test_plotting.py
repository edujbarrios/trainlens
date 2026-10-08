import pytest

from trainlens import plot_training_curves, training_curves

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


def _namespace() -> dict[str, object]:
    return {
        "history": {
            "train_loss": [1.0, 0.7, 0.4],
            "val_loss": [1.1, 0.75, 0.52],
            "train_accuracy": [0.60, 0.76, 0.89],
            "val_accuracy": [0.58, 0.73, 0.84],
        },
        "test_metrics": {"loss": 0.57, "accuracy": 0.82},
    }


def test_training_curves_returns_split_aware_plot_data() -> None:
    curves = training_curves(_namespace())

    assert [(curve.metric, curve.split) for curve in curves] == [
        ("loss", "train"),
        ("loss", "validation"),
        ("loss", "test"),
        ("accuracy", "train"),
        ("accuracy", "validation"),
        ("accuracy", "test"),
    ]
    assert curves[2].values == (0.57,)


def test_plot_training_curves_places_final_test_point_at_last_observation() -> None:
    figure = plot_training_curves(_namespace())

    assert len(figure.axes) == 2
    loss_axis = figure.axes[0]
    assert loss_axis.get_title() == "Loss"
    assert [line.get_label() for line in loss_axis.lines] == ["train", "validation", "test (held-out)"]
    test_line = loss_axis.lines[-1]
    assert tuple(test_line.get_xdata()) == (3,)
    assert tuple(test_line.get_ydata()) == (0.57,)
