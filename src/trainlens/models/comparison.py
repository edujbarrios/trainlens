"""Run comparison models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ComparisonDirection = Literal["improved", "regressed", "unchanged", "new", "removed", "unknown"]
ChangeMagnitude = Literal["none", "small", "material"]


@dataclass(frozen=True)
class MetricComparison:
    """Comparison for one metric across two training runs."""

    name: str
    baseline: float | None
    experiment: float | None
    delta: float | None
    relative_delta: float | None
    direction: ComparisonDirection
    magnitude: ChangeMagnitude

    @property
    def is_actionable(self) -> bool:
        """Whether the metric changed enough to call out in a summary."""

        return self.direction in {"improved", "regressed", "new", "removed"} and (
            self.magnitude == "material" or self.direction in {"new", "removed"}
        )


@dataclass(frozen=True)
class RunComparison:
    """Structured comparison between a baseline and experiment run."""

    baseline_name: str
    experiment_name: str
    metrics: tuple[MetricComparison, ...] = ()
    summary: tuple[str, ...] = ()
    improvements: tuple[MetricComparison, ...] = ()
    regressions: tuple[MetricComparison, ...] = ()
    unchanged: tuple[MetricComparison, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    def has_findings(self) -> bool:
        """Return whether the comparison contains any metric evidence."""

        return bool(self.metrics or self.summary or self.notes)
