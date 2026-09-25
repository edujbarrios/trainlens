import math

import pytest

from trainlens.analyzers.metrics import extract_metric_series, paired_metric
from trainlens.models.metric import MetricSeries


class ScalarTensor:
    def __init__(self, value: float) -> None:
        self.value = value

    def item(self) -> float:
        return self.value


class OneDimensionalArray:
    shape = (3,)

    def __init__(self, values: tuple[float, ...] = (1.0, 0.8, 0.6)) -> None:
        self.values = values

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self):
        return iter(self.values)


class TwoDimensionalArray(OneDimensionalArray):
    shape = (1, 3)


def test_extracts_keras_style_history():
    series = extract_metric_series(
        {"history": {"accuracy": [0.7, 0.8], "val_accuracy": [0.68, 0.75]}}
    )

    train, validation = paired_metric(series, "accuracy")

    assert train is not None
    assert validation is not None
    assert train.last == 0.8
    assert validation.split == "validation"


def test_extracts_trainer_style_log_history():
    series = extract_metric_series(
        {
            "log_history": [
                {"step": 0, "loss": 2.4},
                {"step": 1, "loss": 2.1, "eval_loss": 2.3},
                {"step": 2, "loss": 1.9, "eval_loss": 2.0},
            ]
        }
    )

    train, validation = paired_metric(series, "loss")

    assert train is not None
    assert validation is not None
    assert train.values == (2.4, 2.1, 1.9)
    assert train.steps == (0, 1, 2)
    assert validation.values == (2.3, 2.0)
    assert validation.steps == (1, 2)


def test_invalid_step_metadata_keeps_values_and_steps_aligned():
    series = extract_metric_series(
        {
            "log_history": [
                {"step": 1, "loss": 1.0},
                {"step": "invalid", "loss": 0.9},
                {"step": 3, "loss": 0.8},
            ]
        }
    )["loss"]

    assert series.values == (1.0, 0.9, 0.8)
    assert series.steps == (1, None, 3)
    assert len(series.values) == len(series.steps)


def test_fractional_epochs_are_preserved_without_integer_truncation():
    series = extract_metric_series(
        {
            "epoch_logs": [
                {"epoch": 0.25, "loss": 1.0},
                {"epoch": 0.5, "loss": 0.8},
                {"epoch": 1.0, "loss": 0.6},
            ]
        }
    )["loss"]

    assert series.steps == (0.25, 0.5, 1.0)


def test_metric_series_rejects_misaligned_step_metadata():
    with pytest.raises(ValueError, match="align 1:1"):
        MetricSeries("loss", (1.0, 0.8), steps=(1,))


def test_extracts_pytorch_loop_loss_lists():
    series = extract_metric_series(
        {
            "train_losses": [2.3, 1.8, 1.4],
            "val_losses": [2.4, 1.9, 1.7],
        }
    )

    train, validation = paired_metric(series, "loss")

    assert train is not None
    assert validation is not None
    assert train.name == "train_loss"
    assert train.values == (2.3, 1.8, 1.4)
    assert validation.name == "validation_loss"
    assert validation.values == (2.4, 1.9, 1.7)


def test_extracts_common_one_dimensional_array_like_histories():
    series = extract_metric_series({"train_losses": OneDimensionalArray()})

    assert series["train_loss"].values == (1.0, 0.8, 0.6)


def test_does_not_treat_multidimensional_array_like_values_as_metric_histories():
    series = extract_metric_series({"train_losses": TwoDimensionalArray()})

    assert "train_loss" not in series


def test_extracts_pytorch_epoch_logs():
    series = extract_metric_series(
        {
            "epoch_logs": [
                {"epoch": 1, "train_loss": 2.0, "val_loss": 2.2},
                {"epoch": 2, "train_loss": 1.6, "val_loss": 1.8},
            ]
        }
    )

    train, validation = paired_metric(series, "loss")

    assert train is not None
    assert validation is not None
    assert train.values == (2.0, 1.6)
    assert train.steps == (1.0, 2.0)
    assert validation.values == (2.2, 1.8)
    assert validation.steps == (1.0, 2.0)


def test_extracts_pytorch_lightning_tensor_metrics():
    series = extract_metric_series(
        {
            "callback_metrics": {
                "train_loss": ScalarTensor(1.4),
                "val_loss": ScalarTensor(1.6),
            }
        }
    )

    train, validation = paired_metric(series, "loss")

    assert train is not None
    assert validation is not None
    assert train.last == 1.4
    assert validation.last == 1.6


def test_extracts_tensor_like_values_from_pytorch_logs():
    series = extract_metric_series(
        {
            "train_logs": [
                {"epoch": 1, "loss": ScalarTensor(2.1)},
                {"epoch": 2, "loss": ScalarTensor(1.7)},
            ]
        }
    )

    train, _validation = paired_metric(series, "loss")

    assert train is not None
    assert train.values == (2.1, 1.7)
    assert train.steps == (1.0, 2.0)


def test_normalizes_music_generation_metric_aliases():
    series = extract_metric_series(
        {
            "history": {
                "fad": [4.2, 3.8],
                "eval_clap_score": [0.24, 0.31],
                "eval_stft_loss": [0.72, 0.61],
            }
        }
    )

    assert series["frechet_audio_distance"].last == 3.8
    assert series["validation_clap_score"].split == "validation"
    assert series["validation_stft_loss"].last == 0.61


def test_normalizes_slash_and_dash_metric_names():
    series = extract_metric_series(
        {
            "history": {
                "train/accuracy": [0.7, 0.8],
                "eval-accuracy": [0.68, 0.74],
            }
        }
    )

    train, validation = paired_metric(series, "accuracy")

    assert train is not None
    assert validation is not None
    assert train.name == "train_accuracy"
    assert validation.name == "validation_accuracy"
    assert validation.last == 0.74


def test_omits_non_finite_points_consistently_across_history_formats():
    series = extract_metric_series(
        {
            "metrics": {"loss": float("nan"), "accuracy": float("inf")},
            "history": {"val_loss": [2.0, float("nan"), 1.7]},
            "training_log": [
                {"step": 1, "train_loss": 2.1},
                {"step": 2, "train_loss": float("inf")},
                {"step": 3, "train_loss": 1.7},
            ],
        }
    )

    assert "loss" not in series
    assert "accuracy" not in series
    assert series["validation_loss"].values == (2.0, 1.7)
    assert series["train_loss"].values == (2.1, 1.7)
    assert series["train_loss"].steps == (1, 3)
    assert all(math.isfinite(value) for item in series.values() for value in item.values)
