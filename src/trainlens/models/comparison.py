"""Run comparison models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from trainlens.models.run import RunValue

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
class ParameterChange:
    """One changed configuration value across two runs."""

    name: str
    baseline: RunValue
    experiment: RunValue


@dataclass(frozen=True)
class TrajectoryComparison:
    """Summary of full metric histories across two runs."""

    name: str
    baseline_best: float | None
    experiment_best: float | None
    baseline_observations: int
    experiment_observations: int


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
    parameter_changes: tuple[ParameterChange, ...] = ()
    trajectories: tuple[TrajectoryComparison, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    def has_findings(self) -> bool:
        """Return whether the comparison contains any experiment evidence."""

        return bool(
            self.metrics
            or self.summary
            or self.parameter_changes
            or self.trajectories
            or self.notes
        )

    def to_markdown(self) -> str:
        """Render the comparison as Markdown."""

        from trainlens.comparison import render_run_comparison

        return render_run_comparison(self)

    def _repr_markdown_(self) -> str:
        """Render comparisons automatically in Markdown-capable notebooks."""

        return self.to_markdown()
