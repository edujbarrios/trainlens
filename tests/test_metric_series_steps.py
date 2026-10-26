"""Regression tests for explicit metric time-axis semantics."""

from __future__ import annotations

import pytest

from trainlens.models.metric import MetricSeries


def test_step_aware_slope_uses_actual_irregular_steps() -> None:
    series = MetricSeries(name="loss", values=(1.0, 0.9, 0.5), steps=(1, 2, 10))
    assert series.recent_slope == pytest.approx(-0.25)
    assert series.recent_slope_per_step == pytest.approx(-0.5 / 9)


def test_step_aware_slope_is_unknown_without_complete_steps() -> None:
    assert MetricSeries("loss", (1.0, 0.5)).recent_slope_per_step is None
    assert MetricSeries("loss", (1.0, 0.5), steps=(1, None)).recent_slope_per_step is None
    assert MetricSeries("loss", (1.0, 0.5), steps=(2, 2)).recent_slope_per_step is None


@pytest.mark.parametrize("step", [True, float("nan"), float("inf"), "bad"])
def test_metric_series_rejects_invalid_explicit_step(step: object) -> None:
    with pytest.raises(ValueError, match="finite numbers"):
        MetricSeries(name="loss", values=(1.0,), steps=(step,))
