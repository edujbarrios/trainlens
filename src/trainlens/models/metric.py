"""Metric series primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean


@dataclass(frozen=True)
class MetricPoint:
    """One observed metric value at an optional step."""

    name: str
    value: float
    step: int | None = None
    split: str | None = None


@dataclass(frozen=True)
class MetricSeries:
    """A normalized sequence of metric observations."""

    name: str
    values: tuple[float, ...]
    split: str | None = None
    steps: tuple[int, ...] = field(default_factory=tuple)

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
    def volatility(self) -> float:
        if len(self.values) < 2:
            return 0.0
        avg = mean(self.values)
        return mean(abs(value - avg) for value in self.values)
