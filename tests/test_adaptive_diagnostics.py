"""Synthetic regression tests for scale-aware trend diagnostics."""

import pytest

from trainlens.adaptive import diagnose_trend
from trainlens.models.metric import MetricSeries


@pytest.mark.parametrize("scale", [0.01, 1.0, 1000.0])
def test_classification_is_scale_invariant(scale: float) -> None:
    values = tuple(scale * value for value in (1.0, 0.8, 0.6, 0.4, 0.2))
    result = diagnose_trend(MetricSeries("validation_loss", values))
    assert result is not None
    assert result.state == "improving"


def test_plateau_remains_stable_across_scales() -> None:
    for scale in (0.01, 1.0, 1000.0):
        values = tuple(scale * value for value in (1.0, 1.001, 0.999, 1.0, 1.0))
        result = diagnose_trend(MetricSeries("validation_loss", values))
        assert result is not None
        assert result.state == "stable"


def test_direction_and_unknown_metric() -> None:
    values = (0.2, 0.3, 0.4, 0.5, 0.6)
    assert diagnose_trend(MetricSeries("validation_accuracy", values)).state == "improving"
    assert diagnose_trend(MetricSeries("custom_unknown_metric", values)).state == "changing"


def test_rejects_non_finite_and_insufficient_data() -> None:
    assert diagnose_trend(MetricSeries("loss", (1.0, 0.8))) is None
    assert diagnose_trend(MetricSeries("loss", (1.0, 0.8, float("nan"), 0.6, 0.5))) is None


@pytest.mark.parametrize("kwargs", [
    {"window": 2},
    {"window": True},
    {"relative_tolerance": -0.1},
    {"noise_multiplier": float("inf")},
])
def test_rejects_invalid_configuration(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        diagnose_trend(MetricSeries("loss", (1.0, 0.8, 0.6, 0.4, 0.2)), **kwargs)  # type: ignore[arg-type]
