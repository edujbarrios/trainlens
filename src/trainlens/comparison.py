"""Compare TrainLens training runs."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import TypeAlias

from trainlens.metric_semantics import metric_direction, metric_material_thresholds
from trainlens.models.analysis import AnalysisResult
from trainlens.models.comparison import (
    ChangeMagnitude,
    ComparisonDirection,
    MetricComparison,
    ParameterChange,
    RunComparison,
    TrajectoryComparison,
)
from trainlens.models.metric import MetricSeries
from trainlens.models.run import TrainingRun

RunLike: TypeAlias = AnalysisResult | TrainingRun | Mapping[str, float]

_MATERIAL_RELATIVE_DELTA = 0.05
_MATERIAL_ABSOLUTE_DELTA = 0.01


def compare_runs(
    baseline: RunLike,
    experiment: RunLike,
    *,
    baseline_name: str | None = None,
    experiment_name: str | None = None,
) -> RunComparison:
    """Compare two runs across final metrics, configuration, and trajectories."""

    baseline_metrics = _metrics_from_run(baseline)
    experiment_metrics = _metrics_from_run(experiment)
    metric_names = tuple(sorted(set(baseline_metrics) | set(experiment_metrics)))
    comparisons = tuple(
        _compare_metric(name, baseline_metrics.get(name), experiment_metrics.get(name))
        for name in metric_names
    )
    improvements = tuple(item for item in comparisons if item.direction == "improved")
    regressions = tuple(item for item in comparisons if item.direction == "regressed")
    unchanged = tuple(item for item in comparisons if item.direction == "unchanged")
    parameter_changes = _parameter_changes(baseline, experiment)
    trajectories = _trajectory_comparisons(baseline, experiment)
    return RunComparison(
        baseline_name=baseline_name or _run_name(baseline, fallback="baseline"),
        experiment_name=experiment_name or _run_name(experiment, fallback="experiment"),
        metrics=comparisons,
        summary=_summary(improvements, regressions, comparisons),
        improvements=improvements,
        regressions=regressions,
        unchanged=unchanged,
        parameter_changes=parameter_changes,
        trajectories=trajectories,
        notes=_notes(comparisons),
    )


def render_run_comparison(comparison: RunComparison) -> str:
    """Render a run comparison as Markdown."""

    lines = [
        "## TrainLens Run Comparison",
        "",
        f"**Baseline:** {comparison.baseline_name}",
        f"**Experiment:** {comparison.experiment_name}",
    ]
    if comparison.summary:
        lines.extend(["", "### Summary"])
        lines.extend(f"- {item}" for item in comparison.summary)
    if comparison.parameter_changes:
        lines.extend(
            [
                "",
                "### Configuration changes",
                "| Parameter | Baseline | Experiment |",
                "| --- | --- | --- |",
            ]
        )
        for item in comparison.parameter_changes:
            lines.append(
                f"| {_escape_table_cell(item.name)} | `{item.baseline}` | `{item.experiment}` |"
            )
    if comparison.metrics:
        lines.extend(
            [
                "",
                "### Metric changes",
                "| Metric | Baseline | Experiment | Delta | Relative | Direction | Magnitude |",
                "| --- | ---: | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for item in comparison.metrics:
            lines.append(
                "| "
                f"{_escape_table_cell(item.name)} | "
                f"{_format_optional_float(item.baseline)} | "
                f"{_format_optional_float(item.experiment)} | "
                f"{_format_optional_float(item.delta, signed=True)} | "
                f"{_format_percent(item.relative_delta)} | "
                f"{item.direction} | {item.magnitude} |"
            )
    if comparison.trajectories:
        lines.extend(
            [
                "",
                "### Training trajectories",
                "| Metric | Baseline best | Experiment best | Baseline obs. | Experiment obs. |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for item in comparison.trajectories:
            lines.append(
                f"| {_escape_table_cell(item.name)} | "
                f"{_format_optional_float(item.baseline_best)} | "
                f"{_format_optional_float(item.experiment_best)} | "
                f"{item.baseline_observations} | {item.experiment_observations} |"
            )
    if comparison.notes:
        lines.extend(["", "### Notes"])
        lines.extend(f"- {item}" for item in comparison.notes)
    return "\n".join(lines).strip() + "\n"


def _compare_metric(
    name: str,
    baseline: float | None,
    experiment: float | None,
) -> MetricComparison:
    if baseline is None:
        return MetricComparison(name, None, experiment, None, None, "new", "material")
    if experiment is None:
        return MetricComparison(name, baseline, None, None, None, "removed", "material")
    delta = experiment - baseline
    relative_delta = _relative_delta(baseline, delta)
    magnitude = _magnitude(name, delta, relative_delta)
    direction = _direction(name, delta, magnitude)
    return MetricComparison(
        name=name,
        baseline=baseline,
        experiment=experiment,
        delta=delta,
        relative_delta=relative_delta,
        direction=direction,
        magnitude=magnitude,
    )


def _parameter_changes(
    baseline: RunLike,
    experiment: RunLike,
) -> tuple[ParameterChange, ...]:
    if not isinstance(baseline, TrainingRun) or not isinstance(experiment, TrainingRun):
        return ()
    names = sorted(set(baseline.parameters) | set(experiment.parameters))
    return tuple(
        ParameterChange(name, baseline.parameters.get(name), experiment.parameters.get(name))
        for name in names
        if baseline.parameters.get(name) != experiment.parameters.get(name)
    )


def _trajectory_comparisons(
    baseline: RunLike,
    experiment: RunLike,
) -> tuple[TrajectoryComparison, ...]:
    if not isinstance(baseline, TrainingRun) or not isinstance(experiment, TrainingRun):
        return ()
    baseline_series = {metric.name: metric for metric in baseline.metrics}
    experiment_series = {metric.name: metric for metric in experiment.metrics}
    common = sorted(set(baseline_series) & set(experiment_series))
    return tuple(
        TrajectoryComparison(
            name=name,
            baseline_best=_best_value(name, baseline_series[name]),
            experiment_best=_best_value(name, experiment_series[name]),
            baseline_observations=len(baseline_series[name].values),
            experiment_observations=len(experiment_series[name].values),
        )
        for name in common
        if len(baseline_series[name].values) > 1 or len(experiment_series[name].values) > 1
    )


def _best_value(name: str, series: MetricSeries) -> float | None:
    values = tuple(value for value in series.values if isfinite(value))
    if not values:
        return None
    direction = metric_direction(name)
    if direction == "lower":
        return min(values)
    if direction == "higher":
        return max(values)
    return series.last


def _direction(name: str, delta: float, magnitude: ChangeMagnitude) -> ComparisonDirection:
    if magnitude == "none":
        return "unchanged"
    preferred = metric_direction(name)
    if preferred == "lower":
        return "improved" if delta < 0 else "regressed"
    if preferred == "higher":
        return "improved" if delta > 0 else "regressed"
    return "unknown"


def _magnitude(name: str, delta: float, relative_delta: float | None) -> ChangeMagnitude:
    if abs(delta) < 1e-12:
        return "none"
    custom_relative, custom_absolute = metric_material_thresholds(name)
    relative_threshold = _MATERIAL_RELATIVE_DELTA if custom_relative is None else custom_relative
    absolute_threshold = _MATERIAL_ABSOLUTE_DELTA if custom_absolute is None else custom_absolute
    if relative_delta is not None and abs(relative_delta) >= relative_threshold:
        return "material"
    if abs(delta) >= absolute_threshold:
        return "material"
    return "small"


def _relative_delta(baseline: float, delta: float) -> float | None:
    return None if abs(baseline) < 1e-12 else delta / abs(baseline)


def _summary(
    improvements: tuple[MetricComparison, ...],
    regressions: tuple[MetricComparison, ...],
    comparisons: tuple[MetricComparison, ...],
) -> tuple[str, ...]:
    lines: list[str] = []
    material_improvements = [item for item in improvements if item.magnitude == "material"]
    material_regressions = [item for item in regressions if item.magnitude == "material"]
    material_unknown = [
        item for item in comparisons if item.direction == "unknown" and item.magnitude == "material"
    ]
    if material_improvements:
        lines.append(
            "Material improvement detected in "
            + ", ".join(item.name for item in material_improvements)
            + "."
        )
    if material_regressions:
        lines.append(
            "Material regression detected in "
            + ", ".join(item.name for item in material_regressions)
            + "."
        )
    if material_unknown:
        lines.append(
            "Material change detected in "
            + ", ".join(item.name for item in material_unknown)
            + ", but optimization direction is unknown."
        )
    new_metrics = [item.name for item in comparisons if item.direction == "new"]
    removed_metrics = [item.name for item in comparisons if item.direction == "removed"]
    if new_metrics:
        lines.append(f"New experiment-only metric(s): {', '.join(new_metrics)}.")
    if removed_metrics:
        lines.append(f"Metric(s) missing from experiment: {', '.join(removed_metrics)}.")
    if not lines and comparisons:
        lines.append("No material metric movement detected.")
    if not comparisons:
        lines.append("No comparable metrics were found.")
    return tuple(lines)


def _notes(comparisons: tuple[MetricComparison, ...]) -> tuple[str, ...]:
    unknown = [item.name for item in comparisons if item.direction == "unknown"]
    if not unknown:
        return ()
    return (
        "Some metric directions are unknown because TrainLens does not know whether "
        f"higher or lower is better for: {', '.join(unknown)}.",
    )


def _metrics_from_run(run: RunLike) -> dict[str, float]:
    if isinstance(run, AnalysisResult):
        return _finite_metrics(run.metrics)
    if isinstance(run, TrainingRun):
        return _finite_metrics(
            {metric.name: metric.last for metric in run.metrics if metric.last is not None}
        )
    return _finite_metrics(run)


def _finite_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    finite: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if isfinite(numeric_value):
            finite[str(key)] = numeric_value
    return finite


def _run_name(run: RunLike, *, fallback: str) -> str:
    if isinstance(run, AnalysisResult):
        return run.model_name or fallback
    if isinstance(run, TrainingRun):
        return run.model_name or run.run_id or fallback
    return fallback


def _format_optional_float(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return ""
    prefix = "+" if signed and value > 0 else ""
    return f"{prefix}{value:.4g}"


def _format_percent(value: float | None) -> str:
    return "" if value is None else f"{value:+.1%}"


def _escape_table_cell(value: str) -> str:
    return value.replace("|", r"\|")
