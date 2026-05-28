"""Metric interpretation heuristics."""

from __future__ import annotations

from trainlens.models.analysis import Signal
from trainlens.models.metric import MetricSeries


def detect_overfitting(
    train: MetricSeries | None, validation: MetricSeries | None
) -> Signal | None:
    if train is None or validation is None or train.last is None or validation.last is None:
        return None
    gap = train.last - validation.last
    if gap >= 0.12:
        return Signal(
            title="Possible overfitting",
            detail=f"Training {train.name} is {gap:.3f} higher than validation.",
            severity="warning",
            evidence=(f"train={train.last:.3f}", f"validation={validation.last:.3f}"),
        )
    return None


def detect_validation_instability(series: MetricSeries | None) -> Signal | None:
    if series is None or len(series.values) < 4:
        return None
    if series.volatility > 0.05:
        return Signal(
            title="Validation metric is unstable",
            detail=f"{series.name} moves around noticeably across recent observations.",
            severity="warning",
            evidence=(f"volatility={series.volatility:.3f}",),
        )
    return None


def detect_convergence(series: MetricSeries | None) -> Signal | None:
    if series is None or series.recent_slope is None:
        return None
    if abs(series.recent_slope) <= 0.005 and len(series.values) >= 4:
        return Signal(
            title="Metric appears to have stabilized",
            detail=f"Recent {series.name} changes are small.",
            severity="info",
            evidence=(f"recent_slope={series.recent_slope:.4f}",),
        )
    return None
