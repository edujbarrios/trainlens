"""Structured notebook context for LLM-only reports."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from trainlens.analyzers.metrics import extract_metric_series
from trainlens.introspection import NotebookInspector
from trainlens.models.metric import MetricSeries
from trainlens.models.snapshot import NotebookSnapshot
from trainlens.security import sanitize_value

_MAX_METRIC_POINTS = 12


@dataclass(frozen=True)
class LLMNotebookContext:
    """Notebook evidence prepared for an LLM report."""

    markdown: str
    metrics: dict[str, float]

    def _repr_markdown_(self) -> str:
        """Render the exact outbound context natively in IPython/Jupyter."""

        return self.markdown


def build_llm_notebook_context(
    namespace: Mapping[str, Any],
    *,
    max_metric_points: int = _MAX_METRIC_POINTS,
    include_values: bool = False,
) -> LLMNotebookContext:
    """Render notebook state as evidence, minimizing unrelated literal values by default."""

    if isinstance(max_metric_points, bool) or not isinstance(max_metric_points, int):
        raise TypeError("max_metric_points must be an integer")
    if max_metric_points < 2:
        msg = "max_metric_points must be at least 2 to preserve metric endpoints."
        raise ValueError(msg)

    inspector = NotebookInspector()
    snapshot = inspector.snapshot(namespace)
    metric_namespace = _namespace_with_framework_metrics(snapshot)
    metric_series = extract_metric_series(metric_namespace)
    metric_variable_names = {
        name
        for name, value in metric_namespace.items()
        if extract_metric_series({name: value})
    }
    metrics = {
        name: series.last
        for name, series in sorted(metric_series.items())
        if series.last is not None
    }
    lines = [
        "# TrainLens Notebook Context",
        "",
        "Use this evidence to generate the training report. Do not add facts that are not present.",
        "",
    ]
    if snapshot.variables:
        lines.extend(["## Notebook Variables", ""])
        for variable in snapshot.variables:
            details = [f"type={variable.type_name}"]
            if variable.module:
                details.append(f"module={variable.module}")
            if variable.shape is not None:
                details.append(f"shape={variable.shape}")
            if variable.length is not None:
                details.append(f"length={variable.length}")
            lines.append(f"- `{variable.name}`: " + ", ".join(details))
            if (
                include_values
                and variable.value is not None
                and variable.name not in metric_variable_names
            ):
                lines.append(f"  value: {variable.value!r}")
        lines.append("")
    if metric_series:
        lines.extend(["## Metric Series", ""])
        for name, series in sorted(metric_series.items()):
            lines.append(f"- `{name}`: {_render_metric_series(series, max_metric_points)}")
        lines.append("")
    training_artifacts = [
        artifact for artifact in snapshot.framework_artifacts if artifact.training_parameters
    ]
    if training_artifacts:
        lines.extend(["## Training Parameters", ""])
        for artifact in training_artifacts:
            lines.append(
                f"- `{artifact.variable_name}` ({artifact.type_name}, {artifact.framework})"
            )
            for name, value in sorted(artifact.training_parameters.items()):
                safe_value = sanitize_value(name, value)
                lines.append(f"  - `{name}`: {safe_value!r}")
        lines.append("")
    candidates = inspector.find_models(snapshot)
    if candidates:
        lines.extend(["## Model Candidates", ""])
        for candidate in candidates:
            reasons = ", ".join(candidate.reasons) or "framework match"
            framework = candidate.framework or "unknown framework"
            lines.append(
                f"- `{candidate.variable_name}`: {candidate.type_name}, "
                f"{framework}, confidence={candidate.confidence:.2f}, reasons={reasons}"
            )
    else:
        lines.extend(
            [
                "## Model Candidates",
                "",
                "- No model object was detected. If a string such as `model_name` is present, "
                "treat it only as user-provided context.",
            ]
        )
    return LLMNotebookContext(markdown="\n".join(lines).strip() + "\n", metrics=metrics)


def _render_metric_series(series: MetricSeries, max_metric_points: int) -> str:
    if series.steps:
        return _render_metric_points(series, max_metric_points)
    return _render_metric_values(series.values, max_metric_points)


def _render_metric_points(series: MetricSeries, max_metric_points: int) -> str:
    points = tuple(zip(series.steps, series.values, strict=True))
    if len(points) <= max_metric_points:
        rendered = ", ".join(_format_metric_point(step, value) for step, value in points)
        return f"points=[{rendered}]"
    indices = _sample_indices(len(points), max_metric_points)
    sampled = tuple(points[index] for index in indices)
    rendered_sample = ", ".join(
        _format_metric_point(step, value) for step, value in sampled
    )
    return (
        f"observations={len(points)}, first_step={_format_step(points[0][0])}, "
        f"last_step={_format_step(points[-1][0])}, "
        f"first={series.values[0]:.6g}, last={series.values[-1]:.6g}, "
        f"min={min(series.values):.6g}, max={max(series.values):.6g}, "
        f"ordered_sample=[{rendered_sample}]"
    )


def _format_metric_point(step: int | float | None, value: float) -> str:
    return f"({_format_step(step)}, {value:.6g})"


def _format_step(step: int | float | None) -> str:
    if step is None:
        return "None"
    return f"{step:g}"


def _render_metric_values(values: tuple[float, ...], max_metric_points: int) -> str:
    if len(values) <= max_metric_points:
        rendered = ", ".join(f"{value:.6g}" for value in values)
        return f"[{rendered}]"
    indices = _sample_indices(len(values), max_metric_points)
    sampled = tuple(values[index] for index in indices)
    rendered_sample = ", ".join(f"{value:.6g}" for value in sampled)
    return (
        f"observations={len(values)}, first={values[0]:.6g}, last={values[-1]:.6g}, "
        f"min={min(values):.6g}, max={max(values):.6g}, "
        f"ordered_sample=[{rendered_sample}]"
    )


def _sample_indices(length: int, limit: int) -> tuple[int, ...]:
    last_index = length - 1
    return tuple(round(position * last_index / (limit - 1)) for position in range(limit))


def _namespace_with_framework_metrics(snapshot: NotebookSnapshot) -> dict[str, Any]:
    namespace = dict(snapshot.raw_namespace)
    for artifact in snapshot.framework_artifacts:
        prefix = f"{artifact.variable_name}_{artifact.framework}"
        if artifact.history:
            namespace[f"{prefix}_history"] = artifact.history
        if artifact.log_history:
            namespace[f"{prefix}_log_history"] = artifact.log_history
        if artifact.latest_metrics:
            namespace[f"{prefix}_metrics"] = artifact.latest_metrics
    return namespace
