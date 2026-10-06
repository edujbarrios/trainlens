"""Repeated-run aggregation and controlled parameter-effect summaries."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite, sqrt
from statistics import mean, median, stdev
from typing import Literal

from trainlens.metric_semantics import metric_direction
from trainlens.models.run import RunValue, TrainingRun
from trainlens.run_metrics import final_metric_values

Confidence = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class MetricAggregate:
    """Observed distribution of one final metric across repeated runs."""

    count: int
    mean: float
    stdev: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class RunGroup:
    """Runs sharing a selected parameter configuration."""

    group_by: tuple[str, ...]
    values: tuple[RunValue, ...]
    run_ids: tuple[str, ...]
    metrics: Mapping[str, MetricAggregate]


@dataclass(frozen=True)
class GroupComparison:
    """Conservative comparison between two repeated-run groups."""

    metric: str
    baseline_mean: float
    candidate_mean: float
    delta: float
    improvement: float
    pooled_stdev: float
    signal_to_noise: float | None
    confidence: Confidence
    baseline_count: int
    candidate_count: int


@dataclass(frozen=True)
class ParameterEffect:
    """Observed effect from controlled run pairs differing in one parameter."""

    parameter: str
    from_value: RunValue
    to_value: RunValue
    metric: str
    pairs: int
    median_delta: float
    median_improvement: float
    improvement_rate: float
    confidence: Confidence
    evidence: tuple[str, ...]


def aggregate_runs(
    runs: Sequence[TrainingRun],
    *,
    by: Sequence[str],
    metrics: Sequence[str] | None = None,
) -> tuple[RunGroup, ...]:
    """Aggregate final metrics across repeated runs sharing selected parameters."""

    group_by = tuple(by)
    if not group_by:
        raise ValueError("aggregate_runs requires at least one grouping parameter")
    selected = None if metrics is None else tuple(metrics)
    buckets: dict[tuple[RunValue, ...], list[TrainingRun]] = defaultdict(list)
    for run in runs:
        key = tuple(run.parameters.get(name) for name in group_by)
        buckets[key].append(run)

    groups: list[RunGroup] = []
    for values, members in buckets.items():
        metric_values: dict[str, list[float]] = defaultdict(list)
        for run in members:
            for name, value in _final_metrics(run).items():
                if selected is None or name in selected:
                    metric_values[name].append(value)
        aggregates = {
            name: _aggregate(values_for_metric)
            for name, values_for_metric in sorted(metric_values.items())
            if values_for_metric
        }
        groups.append(
            RunGroup(
                group_by=group_by,
                values=values,
                run_ids=tuple(sorted(run.run_id for run in members)),
                metrics=aggregates,
            )
        )
    return tuple(sorted(groups, key=lambda group: tuple(repr(value) for value in group.values)))


def compare_run_groups(
    baseline: RunGroup,
    candidate: RunGroup,
    *,
    metric: str,
) -> GroupComparison:
    """Compare group means relative to observed between-seed variability."""

    baseline_metric = baseline.metrics.get(metric)
    candidate_metric = candidate.metrics.get(metric)
    if baseline_metric is None or candidate_metric is None:
        raise ValueError(f"metric {metric!r} must exist in both run groups")
    direction = metric_direction(metric)
    if direction is None:
        raise ValueError(f"cannot infer whether {metric!r} should increase or decrease")

    delta = candidate_metric.mean - baseline_metric.mean
    improvement = -delta if direction == "lower" else delta
    pooled = _pooled_stdev(baseline_metric, candidate_metric)
    ratio = None if pooled <= 0.0 else abs(delta) / pooled
    confidence = _comparison_confidence(
        min(baseline_metric.count, candidate_metric.count), ratio, delta
    )
    return GroupComparison(
        metric=metric,
        baseline_mean=baseline_metric.mean,
        candidate_mean=candidate_metric.mean,
        delta=delta,
        improvement=improvement,
        pooled_stdev=pooled,
        signal_to_noise=ratio,
        confidence=confidence,
        baseline_count=baseline_metric.count,
        candidate_count=candidate_metric.count,
    )


def parameter_effects(
    runs: Sequence[TrainingRun],
    *,
    metric: str,
) -> tuple[ParameterEffect, ...]:
    """Summarize controlled pairs that differ in exactly one recorded parameter."""

    direction = metric_direction(metric)
    if direction is None:
        raise ValueError(f"cannot infer whether {metric!r} should increase or decrease")

    observations: dict[tuple[str, str, str], list[tuple[float, str]]] = defaultdict(list)
    values_by_key: dict[tuple[str, str, str], tuple[RunValue, RunValue]] = {}
    used_runs: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    ordered_runs = tuple(sorted(runs, key=lambda run: run.run_id))
    for index, left in enumerate(ordered_runs):
        for right in ordered_runs[index + 1 :]:
            changed = _single_parameter_change(left, right)
            if changed is None:
                continue
            parameter, left_value, right_value = changed
            matched, seed = _matching_seed(left, right)
            if not matched:
                continue
            left_metric = _final_metrics(left).get(metric)
            right_metric = _final_metrics(right).get(metric)
            if left_metric is None or right_metric is None:
                continue
            from_value, to_value, from_metric, to_metric, from_id, to_id = _orient_pair(
                left_value,
                right_value,
                left_metric,
                right_metric,
                left.run_id,
                right.run_id,
            )
            key = (parameter, repr(from_value), repr(to_value))
            if left.run_id in used_runs[key] or right.run_id in used_runs[key]:
                continue
            used_runs[key].update((left.run_id, right.run_id))
            delta = to_metric - from_metric
            seed_suffix = "" if seed is None else f"; seed={seed!r}"
            evidence = (
                f"{from_id}->{to_id}: {metric} {from_metric:.6g}->{to_metric:.6g}"
                f"{seed_suffix}"
            )
            observations[key].append((delta, evidence))
            values_by_key[key] = (from_value, to_value)

    effects: list[ParameterEffect] = []
    for key, records in sorted(observations.items()):
        parameter = key[0]
        from_value, to_value = values_by_key[key]
        deltas = [delta for delta, _ in records]
        improvements = [(-delta if direction == "lower" else delta) for delta in deltas]
        positive = sum(value > 0 for value in improvements)
        effects.append(
            ParameterEffect(
                parameter=parameter,
                from_value=from_value,
                to_value=to_value,
                metric=metric,
                pairs=len(records),
                median_delta=median(deltas),
                median_improvement=median(improvements),
                improvement_rate=positive / len(records),
                confidence=_effect_confidence(improvements),
                evidence=tuple(item for _, item in records),
            )
        )
    return tuple(effects)


def _final_metrics(run: TrainingRun) -> dict[str, float]:
    return final_metric_values(run)


def _matching_seed(left: TrainingRun, right: TrainingRun) -> tuple[bool, RunValue]:
    sentinel = object()
    left_seed: object = left.metadata.get("seed", sentinel)
    right_seed: object = right.metadata.get("seed", sentinel)
    if left_seed is sentinel and right_seed is sentinel:
        return True, None
    if left_seed is sentinel or right_seed is sentinel or left_seed != right_seed:
        return False, None
    if isinstance(left_seed, str | int | float | bool) or left_seed is None:
        return True, left_seed
    return False, None


def _aggregate(values: Sequence[float]) -> MetricAggregate:
    return MetricAggregate(
        count=len(values),
        mean=mean(values),
        stdev=stdev(values) if len(values) > 1 else 0.0,
        minimum=min(values),
        maximum=max(values),
    )


def _pooled_stdev(left: MetricAggregate, right: MetricAggregate) -> float:
    degrees = left.count + right.count - 2
    if degrees <= 0:
        return 0.0
    variance = (
        max(left.count - 1, 0) * left.stdev**2
        + max(right.count - 1, 0) * right.stdev**2
    ) / degrees
    return sqrt(max(variance, 0.0))


def _comparison_confidence(
    minimum_count: int, signal_to_noise: float | None, delta: float
) -> Confidence:
    if delta == 0.0:
        return "low"
    if minimum_count >= 5 and (signal_to_noise is None or signal_to_noise >= 2.0):
        return "high"
    if minimum_count >= 3 and (signal_to_noise is None or signal_to_noise >= 1.0):
        return "medium"
    return "low"


def _single_parameter_change(
    left: TrainingRun, right: TrainingRun
) -> tuple[str, RunValue, RunValue] | None:
    if set(left.parameters) != set(right.parameters):
        return None
    changed = [
        name for name in left.parameters if left.parameters[name] != right.parameters[name]
    ]
    if len(changed) != 1:
        return None
    name = changed[0]
    return name, left.parameters[name], right.parameters[name]


def _orient_pair(
    left_value: RunValue,
    right_value: RunValue,
    left_metric: float,
    right_metric: float,
    left_id: str,
    right_id: str,
) -> tuple[RunValue, RunValue, float, float, str, str]:
    if repr(left_value) <= repr(right_value):
        return left_value, right_value, left_metric, right_metric, left_id, right_id
    return right_value, left_value, right_metric, left_metric, right_id, left_id


def _effect_confidence(improvements: Sequence[float]) -> Confidence:
    count = len(improvements)
    if count == 0:
        return "low"
    consistency = max(
        sum(value > 0 for value in improvements),
        sum(value < 0 for value in improvements),
    ) / count
    if count >= 5 and consistency >= 0.8:
        return "high"
    if count >= 3 and consistency >= 2 / 3:
        return "medium"
    return "low"
