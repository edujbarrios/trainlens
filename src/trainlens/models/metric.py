"""Metric series primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from statistics import mean


@dataclass(frozen=True)
class MetricPoint:
    """One observed metric value at an optional step."""

    name: str
    value: float
    step: int | float | None = None
    split: str | None = None


@dataclass(frozen=True)
class MetricSeries:
    """A normalized sequence of metric observations.

    When ``steps`` is present it is aligned 1:1 with ``values``. Unknown or invalid
    step metadata is represented explicitly as ``None`` rather than silently
    shifting later observations onto the wrong step.
    """

    name: str
    values: tuple[float, ...]
    split: str | None = None
    steps: tuple[int | float | None, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.steps and len(self.steps) != len(self.values):
            raise ValueError("MetricSeries steps must align 1:1 with values")
        for step in self.steps:
            if step is None:
                continue
            if isinstance(step, bool) or not isinstance(step, int | float) or not isfinite(step):
                raise ValueError("MetricSeries steps must be finite numbers or None")

    @property
    def first(self) -> float | None:
        return self.values[0] if self.values else None

    @property
    def last(self) -> float | None:
        return self.values[-1] if self.values else None

    @property
    def delta(self) -> float | None:
        if len(self.values) < 2:
            return None
        return self.values[-1] - self.values[0]

    @property
    def recent_slope(self) -> float | None:
        if len(self.values) < 3:
            return self.delta
        tail = self.values[-3:]
        return (tail[-1] - tail[0]) / 2

    @property
    def recent_slope_per_step(self) -> float | None:
        """Slope over the recent actual steps; unavailable for incomplete metadata."""
        if not self.steps or len(self.values) < 2:
            return None
        tail = min(3, len(self.values))
        positions = self.steps[-tail:]
        steps = tuple(position for position in positions if position is not None)
        if len(steps) != len(positions):
            return None
        if any(right <= left for left, right in zip(steps, steps[1:], strict=False)):
            return None
        distance = steps[-1] - steps[0]
        return (self.values[-1] - self.values[-tail]) / distance

    @property
    def volatility(self) -> float:
        if len(self.values) < 2:
            return 0.0
        avg = mean(self.values)
        return mean(abs(value - avg) for value in self.values)
