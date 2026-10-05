"""Deterministic checkpoint selection from validation trajectories."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Literal, TypeAlias

from trainlens.metric_semantics import MetricDirection, metric_direction
from trainlens.models.metric import MetricSeries
from trainlens.models.run import TrainingRun

CheckpointDirection = Literal["min", "max"]
CheckpointSource: TypeAlias = MetricSeries | TrainingRun | Mapping[str, Sequence[float]]


@dataclass(frozen=True)
class CheckpointSelection:
    """Deterministic selection and early-stopping evidence for one metric trajectory."""

    metric: str
    direction: MetricDirection
    best_index: int
    best_step: int | float
    best_value: float
    last_value: float
    overfit_after: int | float | None
    suggested_patience: int
    degradation_from_best: float
    observations: int


def select_checkpoint(
    source: CheckpointSource,
    *,
    metric: str = "validation_loss",
    direction: CheckpointDirection | None = None,
    min_delta: float = 0.0,
) -> CheckpointSelection:
    """Select the best non-test checkpoint and summarize late-run degradation.

    ``min_delta`` controls what counts as a meaningful improvement/degradation for the
    patience heuristic; it does not move the actual best checkpoint away from the observed
    optimum. Test metrics are rejected so held-out evidence cannot accidentally drive tuning.
    """

    if isinstance(min_delta, bool) or not isinstance(min_delta, int | float):
        raise TypeError("min_delta must be numeric")
    if not isfinite(float(min_delta)) or min_delta < 0:
        raise ValueError("min_delta must be finite and non-negative")

    series = _series_from_source(source, metric)
    _reject_test_selection(series, metric)
    optimization = _resolve_direction(metric, direction)
    values, steps = _finite_aligned_values(series)
    if not values:
        raise ValueError(f"metric {metric!r} has no finite observations")

    best_index = _best_index(values, optimization)
    best_value = values[best_index]
    overfit_index = _first_meaningful_degradation(
        values, best_index, optimization, float(min_delta)
    )
    last_value = values[-1]
    degradation = (
        last_value - best_value
        if optimization == "lower"
        else best_value - last_value
    )
    return CheckpointSelection(
        metric=metric,
        direction=optimization,
        best_index=best_index,
        best_step=steps[best_index],
        best_value=best_value,
        last_value=last_value,
        overfit_after=None if overfit_index is None else steps[overfit_index],
        suggested_patience=_suggested_patience(values, optimization, float(min_delta)),
        degradation_from_best=max(0.0, degradation),
        observations=len(values),
    )


def _series_from_source(source: CheckpointSource, metric: str) -> MetricSeries:
    if isinstance(source, MetricSeries):
        if source.name != metric:
            raise ValueError(
                f"MetricSeries contains {source.name!r}, not requested metric {metric!r}"
            )
        return source
    if isinstance(source, TrainingRun):
        for series in source.metrics:
            if series.name == metric:
                return series
        raise ValueError(f"training run does not contain metric {metric!r}")

    raw = source.get(metric)
    if raw is None:
        raise ValueError(f"mapping does not contain metric {metric!r}")
    values: list[float] = []
    for value in raw:
        if isinstance(value, bool):
            continue
        numeric = float(value)
        values.append(numeric)
    return MetricSeries(name=metric, values=tuple(values))


def _reject_test_selection(series: MetricSeries, metric: str) -> None:
    normalized = metric.lower().replace("-", "_").replace("/", "_")
    if series.split == "test" or normalized == "test" or normalized.startswith("test_"):
        raise ValueError(
            "checkpoint selection must use training/validation evidence, not test metrics"
        )


def _resolve_direction(
    metric: str, explicit: CheckpointDirection | None
) -> MetricDirection:
    if explicit == "min":
        return "lower"
    if explicit == "max":
        return "higher"
    inferred = metric_direction(metric)
    if inferred is None:
        raise ValueError(
            f"cannot infer whether {metric!r} should increase or decrease; pass direction"
        )
    return inferred


def _finite_aligned_values(
    series: MetricSeries,
) -> tuple[list[float], list[int | float]]:
    values: list[float] = []
    steps: list[int | float] = []
    for index, value in enumerate(series.values):
        if not isfinite(value):
            continue
        step = series.steps[index] if series.steps else index
        if step is None:
            step = index
        values.append(value)
        steps.append(step)
    return values, steps


def _best_index(values: Sequence[float], direction: MetricDirection) -> int:
    best = min(values) if direction == "lower" else max(values)
    return next(index for index, value in enumerate(values) if value == best)


def _is_improvement(
    candidate: float,
    incumbent: float,
    direction: MetricDirection,
    min_delta: float,
) -> bool:
    if direction == "lower":
        return candidate < incumbent - min_delta
    return candidate > incumbent + min_delta


def _first_meaningful_degradation(
    values: Sequence[float],
    best_index: int,
    direction: MetricDirection,
    min_delta: float,
) -> int | None:
    best = values[best_index]
    for index in range(best_index + 1, len(values)):
        value = values[index]
        degraded = (
            value > best + min_delta
            if direction == "lower"
            else value < best - min_delta
        )
        if degraded:
            return index
    return None


def _suggested_patience(
    values: Sequence[float], direction: MetricDirection, min_delta: float
) -> int:
    if len(values) < 2:
        return 1
    incumbent = values[0]
    last_improvement = 0
    gaps: list[int] = []
    for index, value in enumerate(values[1:], start=1):
        if _is_improvement(value, incumbent, direction, min_delta):
            gaps.append(index - last_improvement)
            incumbent = value
            last_improvement = index
    if not gaps:
        return max(1, min(3, len(values) - 1))
    return max(1, max(gaps))
