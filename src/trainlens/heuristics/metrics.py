"""Metric interpretation heuristics."""

from __future__ import annotations

from trainlens.metric_semantics import metric_direction
from trainlens.models.analysis import Signal
from trainlens.models.metric import MetricSeries


def detect_overfitting(
    train: MetricSeries | None, validation: MetricSeries | None
) -> Signal | None:
    """Detect a material train/validation generalization gap.

    The rule is direction-aware, so it works for both higher-is-better metrics
    such as accuracy/F1 and lower-is-better metrics such as loss/perplexity.
    """

    if train is None or validation is None or train.last is None or validation.last is None:
        return None
    direction = metric_direction(validation.name) or metric_direction(train.name)
    if direction is None:
        return None
    gap = train.last - validation.last if direction == "higher" else validation.last - train.last
    if gap <= 0 or not _material_gap(train.last, validation.last, gap, conservative=True):
        return None
    metric = _base_metric_name(validation.name)
    return Signal(
        title="Possible overfitting",
        detail=(
            f"Training {metric} materially outperforms validation {metric}; "
            "the run may be fitting the training split better than it generalizes."
        ),
        severity="warning",
        evidence=(
            f"train_{metric}={train.last:.3f}",
            f"validation_{metric}={validation.last:.3f}",
            f"gap={gap:.3f}",
        ),
    )


def detect_test_shift(
    validation: MetricSeries | None, test: MetricSeries | None
) -> Signal | None:
    """Flag a material validation-to-test degradation without asserting its cause."""

    if validation is None or test is None or validation.last is None or test.last is None:
        return None
    direction = metric_direction(validation.name) or metric_direction(test.name)
    if direction is None:
        return None
    degradation = (
        validation.last - test.last if direction == "higher" else test.last - validation.last
    )
    if degradation <= 0 or not _material_gap(
        validation.last, test.last, degradation, conservative=False
    ):
        return None
    metric = _base_metric_name(validation.name)
    return Signal(
        title="Test split underperforms validation",
        detail=(
            f"Held-out test {metric} is materially worse than validation {metric}. "
            "Check for validation over-selection, split mismatch, or distribution shift."
        ),
        severity="warning",
        evidence=(
            f"validation_{metric}={validation.last:.3f}",
            f"test_{metric}={test.last:.3f}",
            f"degradation={degradation:.3f}",
        ),
    )


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


def _material_gap(first: float, second: float, gap: float, *, conservative: bool) -> bool:
    bounded = 0.0 <= first <= 1.0 and 0.0 <= second <= 1.0
    if bounded:
        return gap >= (0.12 if conservative else 0.05)
    scale = max(abs(first), abs(second), 1e-12)
    relative = gap / scale
    return gap >= 0.02 and relative >= (0.20 if conservative else 0.10)


def _base_metric_name(name: str) -> str:
    normalized = name.lower()
    for prefix in ("train_", "training_", "validation_", "val_", "eval_", "test_"):
        if normalized.startswith(prefix):
            return normalized.removeprefix(prefix)
    return normalized
