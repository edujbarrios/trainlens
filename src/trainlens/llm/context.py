"""Structured notebook context for LLM-only reports."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from trainlens.analyzers.metrics import extract_metric_series
from trainlens.introspection import NotebookInspector


@dataclass(frozen=True)
class LLMNotebookContext:
    """Notebook evidence prepared for an LLM report."""

    markdown: str
    metrics: dict[str, float]


def build_llm_notebook_context(namespace: Mapping[str, Any]) -> LLMNotebookContext:
    """Render notebook state as evidence, without heuristic findings."""

    inspector = NotebookInspector()
    snapshot = inspector.snapshot(namespace)
    metric_series = extract_metric_series(snapshot.raw_namespace)
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
            if variable.value is not None:
                lines.append(f"  value: {variable.value!r}")
        lines.append("")
    if metric_series:
        lines.extend(["## Metric Series", ""])
        for name, series in sorted(metric_series.items()):
            values = ", ".join(f"{value:.6g}" for value in series.values)
            lines.append(f"- `{name}`: [{values}]")
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
