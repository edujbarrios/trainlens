"""Structured notebook context for LLM-only reports."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from trainlens.analysis_config import AnalysisConfig
from trainlens.analyzers.metrics import extract_metric_series
from trainlens.introspection import NotebookInspector
from trainlens.introspection.selection import framework_source_for_model
from trainlens.models.analysis import AnalysisResult
from trainlens.models.metric import MetricSeries
from trainlens.models.snapshot import NotebookSnapshot
from trainlens.security import sanitize_value
from trainlens.training_profile import inspect_training_profile

_MAX_METRIC_POINTS = 12


@dataclass(frozen=True)
class LLMNotebookContext:
    """Notebook evidence prepared for an LLM report."""

    markdown: str
    metrics: dict[str, float]

    def _repr_markdown_(self) -> str:
        return self.markdown


def build_llm_notebook_context(
    namespace: Mapping[str, Any],
    *,
    max_metric_points: int = _MAX_METRIC_POINTS,
    include_values: bool = False,
    analysis_config: AnalysisConfig | None = None,
    deterministic_result: AnalysisResult | None = None,
) -> LLMNotebookContext:
    """Capture and render notebook state as bounded LLM evidence."""

    snapshot = NotebookInspector().snapshot(namespace)
    return build_llm_notebook_context_from_snapshot(
        snapshot,
        max_metric_points=max_metric_points,
        include_values=include_values,
        analysis_config=analysis_config,
        deterministic_result=deterministic_result,
    )


def build_llm_notebook_context_from_snapshot(
    snapshot: NotebookSnapshot,
    *,
    max_metric_points: int = _MAX_METRIC_POINTS,
    include_values: bool = False,
    analysis_config: AnalysisConfig | None = None,
    deterministic_result: AnalysisResult | None = None,
) -> LLMNotebookContext:
    """Render an existing notebook snapshot without inspecting live state again."""

    if isinstance(max_metric_points, bool) or not isinstance(max_metric_points, int):
        raise TypeError("max_metric_points must be an integer")
    if max_metric_points < 2:
        raise ValueError("max_metric_points must be at least 2 to preserve metric endpoints.")

    inspector = NotebookInspector()
    candidates = inspector.find_models(snapshot)
    model_ref = _selected_model_ref(snapshot, candidates, analysis_config)
    trainer = _selected_trainer(snapshot, model_ref, analysis_config)
    training_profile = inspect_training_profile(
        model_ref,
        trainer=trainer,
        namespace=snapshot.raw_namespace,
    )
    metric_namespace = _namespace_with_framework_metrics(snapshot)
    if analysis_config is not None and analysis_config.metrics is not None:
        metric_namespace["trainlens_explicit_metrics"] = analysis_config.metrics
    metric_series = extract_metric_series(metric_namespace)
    metric_variable_names = {
        name for name, value in metric_namespace.items() if extract_metric_series({name: value})
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
        "Treat the TrainLens deterministic findings below as conclusions already derived by "
        "the local analyzer; explain them and their evidence rather than silently replacing them.",
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
            if include_values and variable.value is not None and variable.name not in metric_variable_names:
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
            lines.append(f"- `{artifact.variable_name}` ({artifact.type_name}, {artifact.framework})")
            for name, value in sorted(artifact.training_parameters.items()):
                lines.append(f"  - `{name}`: {sanitize_value(name, value)!r}")
        lines.append("")
    if (
        training_profile.parameters
        or training_profile.trainable_components
        or training_profile.frozen_components
        or training_profile.observations
    ):
        lines.extend(["## Training Profile", ""])
        lines.append(f"- strategy: `{training_profile.strategy}`")
        if training_profile.trainable_components:
            lines.append(
                "- trainable components: "
                + ", ".join(f"`{name}`" for name in training_profile.trainable_components)
            )
        if training_profile.frozen_components:
            lines.append(
                "- frozen components: "
                + ", ".join(f"`{name}`" for name in training_profile.frozen_components)
            )
        for name, value in sorted(training_profile.parameters.items()):
            lines.append(f"- `{name}`: {sanitize_value(name, value)!r}")
        lines.append("")
    if candidates:
        lines.extend(["## Model Candidates", ""])
        for candidate in candidates:
            reasons = ", ".join(candidate.reasons) or "framework match"
            framework = candidate.framework or "unknown framework"
            selected = ", selected=true" if candidate.object_ref is model_ref else ""
            lines.append(
                f"- `{candidate.variable_name}`: {candidate.type_name}, {framework}, "
                f"confidence={candidate.confidence:.2f}, reasons={reasons}{selected}"
            )
        lines.append("")
    else:
        lines.extend(
            [
                "## Model Candidates",
                "",
                "- No model object was detected. If a string such as `model_name` is present, "
                "treat it only as user-provided context.",
                "",
            ]
        )
    if deterministic_result is not None:
        lines.extend(_render_deterministic_findings(deterministic_result))
    return LLMNotebookContext(markdown="\n".join(lines).strip() + "\n", metrics=metrics)


def _render_deterministic_findings(result: AnalysisResult) -> list[str]:
    lines = ["## TrainLens Deterministic Findings", ""]
    if result.summary:
        lines.append("### Summary")
        lines.extend(f"- {item}" for item in result.summary)
        lines.append("")
    if result.metrics:
        lines.append("### Final metrics")
        lines.extend(f"- `{name}`: {value:.6g}" for name, value in sorted(result.metrics.items()))
        lines.append("")
    if result.signals:
        lines.append("### Signals")
        for signal in result.signals:
            lines.append(f"- [{signal.severity}] {signal.title}: {signal.detail}")
            for evidence in signal.evidence:
                lines.append(f"  - evidence: {evidence}")
            for ref in signal.evidence_refs:
                location = ref.source
                if ref.metric is not None:
                    location += f", metric={ref.metric}"
                if ref.start_step is not None or ref.end_step is not None:
                    location += f", steps={ref.start_step}..{ref.end_step}"
                lines.append(f"  - provenance: {location}: {ref.detail}")
        lines.append("")
    if result.recommendations:
        lines.append("### Recommendations")
        for recommendation in result.recommendations:
            source = f", source={recommendation.source}" if recommendation.source else ""
            lines.append(
                f"- {recommendation.action} (confidence={recommendation.confidence:.0%}{source})"
            )
            lines.append(f"  - rationale: {recommendation.rationale}")
            for evidence in recommendation.evidence:
                lines.append(f"  - evidence: {evidence}")
        lines.append("")
    return lines


def _selected_model_ref(snapshot: NotebookSnapshot, candidates: list[Any], config: AnalysisConfig | None) -> object | None:
    if config is not None and config.model is not None:
        if isinstance(config.model, str):
            if config.model not in snapshot.raw_namespace:
                raise ValueError(f"model variable {config.model!r} was not found in the notebook snapshot")
            return snapshot.raw_namespace[config.model]
        return config.model
    if config is not None and config.strict and len(candidates) > 1:
        names = ", ".join(candidate.variable_name for candidate in candidates)
        raise ValueError(
            "multiple model candidates detected; select one explicitly with "
            f"AnalysisConfig(model=...). Candidates: {names}"
        )
    return candidates[0].object_ref if candidates else _first_artifact_model_ref(snapshot)


def _selected_trainer(
    snapshot: NotebookSnapshot,
    model_ref: object | None,
    config: AnalysisConfig | None,
) -> object | None:
    if config is not None and config.trainer is not None:
        if isinstance(config.trainer, str):
            if config.trainer not in snapshot.raw_namespace:
                raise ValueError(
                    f"trainer variable {config.trainer!r} was not found in the notebook snapshot"
                )
            return snapshot.raw_namespace[config.trainer]
        return config.trainer
    return framework_source_for_model(snapshot, "huggingface", model_ref)


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
    rendered_sample = ", ".join(_format_metric_point(step, value) for step, value in sampled)
    return (
        f"observations={len(points)}, first_step={_format_step(points[0][0])}, "
        f"last_step={_format_step(points[-1][0])}, first={series.values[0]:.6g}, "
        f"last={series.values[-1]:.6g}, min={min(series.values):.6g}, "
        f"max={max(series.values):.6g}, ordered_sample=[{rendered_sample}]"
    )


def _format_metric_point(step: int | float | None, value: float) -> str:
    return f"({_format_step(step)}, {value:.6g})"


def _format_step(step: int | float | None) -> str:
    if step is None:
        return "None"
    return f"{step:g}"


def _render_metric_values(values: tuple[float, ...], max_metric_points: int) -> str:
    if len(values) <= max_metric_points:
        return "[" + ", ".join(f"{value:.6g}" for value in values) + "]"
    indices = _sample_indices(len(values), max_metric_points)
    sampled = tuple(values[index] for index in indices)
    rendered_sample = ", ".join(f"{value:.6g}" for value in sampled)
    return (
        f"observations={len(values)}, first={values[0]:.6g}, last={values[-1]:.6g}, "
        f"min={min(values):.6g}, max={max(values):.6g}, ordered_sample=[{rendered_sample}]"
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


def _first_artifact_model_ref(snapshot: NotebookSnapshot) -> object | None:
    for artifact in snapshot.framework_artifacts:
        if artifact.model_ref is not None:
            return cast(object, artifact.model_ref)
    return None
