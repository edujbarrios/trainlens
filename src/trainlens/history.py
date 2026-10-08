"""Deterministic, anomaly-aware sketches of long metric trajectories.

The full metric history stays local. Only a bounded sketch is intended for an
LLM/agent context; callers can always inspect the original MetricSeries.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from trainlens.models.metric import MetricSeries


@dataclass(frozen=True)
class HistoryDigest:
    """A bounded, ordered view of a complete metric history.

    Positions are 1-based observation numbers, not inferred training steps.
    Explicit steps, including unknown steps, are preserved independently.
    """

    observations: int
    first: float
    last: float
    minimum: float
    maximum: float
    minimum_position: int
    maximum_position: int
    sampled_positions: tuple[int, ...]
    sampled_values: tuple[float, ...]
    sampled_steps: tuple[int | float | None, ...]


def summarize_metric_history(
    series: MetricSeries, *, max_points: int = 12
) -> HistoryDigest:
    """Keep endpoints, extrema and the most informative changes of direction.

    Uses a deterministic piecewise-linear error sketch, not uniform sampling.
    For budgets of four or more, both global extrema are always represented.
    Complexity is O(observations * max_points) for typical small LLM budgets.
    Nothing mutates or discards the source history.
    """

    if isinstance(max_points, bool) or not isinstance(max_points, int):
        raise TypeError("max_points must be an integer")
    if max_points < 2:
        raise ValueError("max_points must be at least 2")
    values = series.values
    if not values:
        raise ValueError("cannot summarize an empty metric history")
    if not all(isfinite(value) for value in values):
        raise ValueError("metric history must contain only finite values")

    minimum_index = min(range(len(values)), key=values.__getitem__)
    maximum_index = max(range(len(values)), key=values.__getitem__)
    indices = _salient_indices(values, max_points, minimum_index, maximum_index)
    return HistoryDigest(
        observations=len(values),
        first=values[0],
        last=values[-1],
        minimum=values[minimum_index],
        maximum=values[maximum_index],
        minimum_position=minimum_index + 1,
        maximum_position=maximum_index + 1,
        sampled_positions=tuple(index + 1 for index in indices),
        sampled_values=tuple(values[index] for index in indices),
        sampled_steps=tuple(
            series.steps[index] if series.steps else None for index in indices
        ),
    )


def _salient_indices(
    values: tuple[float, ...],
    limit: int,
    minimum_index: int,
    maximum_index: int,
) -> tuple[int, ...]:
    length = len(values)
    if length <= limit:
        return tuple(range(length))
    chosen = {0, length - 1}
    if limit >= 4:
        chosen.update((minimum_index, maximum_index))
    elif limit == 3:
        chosen.add(
            max(
                (minimum_index, maximum_index),
                key=lambda i: (
                    abs(values[i] - (values[0] + (values[-1] - values[0]) * i / (length - 1))),
                    -i,
                ),
            )
        )
    scale = max(max(values) - min(values), 1.0)
    tolerance = scale * 1e-12
    while len(chosen) < limit:
        ordered = sorted(chosen)
        best_index = -1
        best_error = tolerance
        for left, right in zip(ordered, ordered[1:], strict=False):
            if right - left < 2:
                continue
            slope = (values[right] - values[left]) / (right - left)
            for index in range(left + 1, right):
                error = abs(values[index] - (values[left] + slope * (index - left)))
                if error > best_error:
                    best_error, best_index = error, index
        if best_index < 0:
            left, right = max(
                zip(ordered, ordered[1:], strict=False), key=lambda pair: pair[1] - pair[0]
            )
            if right - left < 2:
                break
            best_index = (left + right) // 2
        chosen.add(best_index)
    return tuple(sorted(chosen))
