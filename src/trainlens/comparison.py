"""Compare TrainLens training runs."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import TypeAlias

from trainlens.models.analysis import AnalysisResult
from trainlens.models.comparison import (
    ChangeMagnitude,
    ComparisonDirection,
    MetricComparison,
    RunComparison,
)
from trainlens.models.run import TrainingRun

RunLike: TypeAlias = AnalysisResult | TrainingRun | Mapping[str, float]

_LOWER_IS_BETTER = (
    "loss",
    "error",
    "perplexity",
    "wer",
    "cer",
    "latency",
    "fad",
    "frechet",
)
_HIGHER_IS_BETTER = (
    "accuracy",
    "acc",
    "auc",
    "f1",
    "precision",
    "recall",
    "score",
    "map",
    "ndcg",
)
_MATERIAL_RELATIVE_DELTA = 0.05
_MATERIAL_ABSOLUTE_DELTA = 0.01


def compare_runs(
    baseline: RunLike,
    experiment: RunLike,
    *,
    baseline_name: str | None = None,
    experiment_name: str | None = None,
) -> RunComparison:
    """Compare two runs and report metric improvements/regressions."""

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
    notes = _notes(comparisons)
    return RunComparison(
        baseline_name=baseline_name or _run_name(baseline, fallback="baseline"),
        experiment_name=experiment_name or _run_name(experiment, fallback="experiment"),
        metrics=comparisons,
        summary=_summary(improvements, regressions, comparisons),
        improvements=improvements,
        regressions=regressions,
        unchanged=unchanged,
        notes=notes,
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
        lines.append("")
        lines.append("### Summary")
        lines.extend(f"- {item}" for item in comparison.summary)
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
                f"{item.name} | "
                f"{_format_optional_float(item.baseline)} | "
                f"{_format_optional_float(item.experiment)} | "
                f"{_format_optional_float(item.delta, signed=True)} | "
                f"{_format_percent(item.relative_delta)} | "
                f"{item.direction} | "
                f"{item.magnitude} |"
            )
    if comparison.notes:
        lines.append("")
        lines.append("### Notes")
        lines.extend(f"- {item}" for item in comparison.notes)
    return "\n".join(lines).strip() + "\n"


def _compare_metric(
    name: str,
    baseline: float | None,
    experiment: float | None,
) -> MetricComparison:
    if baseline is None:
        return MetricComparison(
            name=name,
            baseline=None,
            experiment=experiment,
            delta=None,
            relative_delta=None,
            direction="new",
            magnitude="material",
        )
    if experiment is None:
        return MetricComparison(
            name=name,
            baseline=baseline,
            experiment=None,
            delta=None,
            relative_delta=None,
            direction="removed",
            magnitude="material",
        )
    delta = experiment - baseline
    relative_delta = _relative_delta(baseline, delta)
    magnitude = _magnitude(delta, relative_delta)
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


def _direction(name: str, delta: float, magnitude: ChangeMagnitude) -> ComparisonDirection:
    if magnitude == "none":
        return "unchanged"
    tokens = set(re.findall(r"[a-z0-9]+", name.lower()))
    if tokens.intersection(_LOWER_IS_BETTER):
        return "improved" if delta < 0 else "regressed"
    if tokens.intersection(_HIGHER_IS_BETTER):
        return "improved" if delta > 0 else "regressed"
    return "unknown"


def _magnitude(delta: float, relative_delta: float | None) -> ChangeMagnitude:
    if abs(delta) < 1e-12:
        return "none"
    if relative_delta is not None and abs(relative_delta) >= _MATERIAL_RELATIVE_DELTA:
        return "material"
    if abs(delta) >= _MATERIAL_ABSOLUTE_DELTA:
        return "material"
    return "small"


def _relative_delta(baseline: float, delta: float) -> float | None:
    if abs(baseline) < 1e-12:
        return None
    return delta / abs(baseline)


def _summary(
    improvements: tuple[MetricComparison, ...],
    regressions: tuple[MetricComparison, ...],
    comparisons: tuple[MetricComparison, ...],
) -> tuple[str, ...]:
    lines: list[str] = []
    material_improvements = [item for item in improvements if item.magnitude == "material"]
    material_regressions = [item for item in regressions if item.magnitude == "material"]
    if material_improvements:
        names = ", ".join(item.name for item in material_improvements)
        lines.append(f"Material improvement detected in {names}.")
    if material_regressions:
        names = ", ".join(item.name for item in material_regressions)
        lines.append(f"Material regression detected in {names}.")
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
        return dict(run.metrics)
    if isinstance(run, TrainingRun):
        return {metric.name: metric.last for metric in run.metrics if metric.last is not None}
    return {str(key): float(value) for key, value in run.items()}


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
    if value is None:
        return ""
    return f"{value:+.1%}"
