"""Portable TrainLens run capture and JSON persistence."""

from __future__ import annotations

import json
from math import isfinite
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trainlens.models.analysis import AnalysisResult
from trainlens.models.metric import MetricSeries
from trainlens.models.run import RunValue, TrainingRun


def training_run_from_analysis(
    result: AnalysisResult,
    *,
    run_id: str | None = None,
    parameters: Mapping[str, RunValue] | None = None,
    metadata: Mapping[str, RunValue] | None = None,
    notes: tuple[str, ...] = (),
) -> TrainingRun:
    """Create a portable run from a deterministic analysis result."""

    metrics = (
        tuple(result.metric_series[name] for name in sorted(result.metric_series))
        if result.metric_series
        else tuple(
            MetricSeries(name=name, values=(value,))
            for name, value in sorted(result.metrics.items())
        )
    )
    kwargs: dict[str, Any] = {
        "model_name": result.model_name,
        "framework": result.framework,
        "metrics": metrics,
        "parameters": dict(parameters or {}),
        "metadata": dict(metadata or {}),
        "notes": notes,
    }
    if run_id is not None:
        kwargs["run_id"] = run_id
    return TrainingRun(**kwargs)


def save_run(run: TrainingRun, path: str | Path) -> Path:
    """Serialize one training run to a human-readable JSON document."""

    destination = Path(path)
    destination.write_text(
        json.dumps(_run_to_dict(run), allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def load_run(path: str | Path) -> TrainingRun:
    """Load a training run previously written by :func:`save_run`."""

    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("TrainLens run JSON must contain an object")
    return _run_from_dict(raw)


def _run_to_dict(run: TrainingRun) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": run.run_id,
        "created_at": run.created_at.astimezone(UTC).isoformat(),
        "model_name": run.model_name,
        "framework": run.framework,
        "parameters": dict(run.parameters),
        "metadata": dict(run.metadata),
        "notes": list(run.notes),
        "metrics": [
            {
                "name": metric.name,
                "values": list(metric.values),
                "split": metric.split,
                "steps": list(metric.steps),
            }
            for metric in run.metrics
        ],
    }


def _run_from_dict(raw: dict[str, Any]) -> TrainingRun:
    version = raw.get("schema_version")
    if version != 1:
        raise ValueError(f"unsupported TrainLens run schema version: {version!r}")
    metrics_raw = raw.get("metrics", [])
    if not isinstance(metrics_raw, list):
        raise ValueError("TrainLens run metrics must be a list")
    metrics = tuple(_metric_from_dict(item) for item in metrics_raw)
    created_at_raw = raw.get("created_at")
    if not isinstance(created_at_raw, str):
        raise ValueError("TrainLens run created_at must be an ISO timestamp")
    created_at = datetime.fromisoformat(created_at_raw)
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    parameters = _simple_mapping(raw.get("parameters", {}), "parameters")
    metadata = _simple_mapping(raw.get("metadata", {}), "metadata")
    notes_raw = raw.get("notes", [])
    if not isinstance(notes_raw, list) or not all(isinstance(item, str) for item in notes_raw):
        raise ValueError("TrainLens run notes must be a list of strings")
    run_id = raw.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("TrainLens run_id must be a non-empty string")
    model_name = raw.get("model_name")
    framework = raw.get("framework")
    if model_name is not None and not isinstance(model_name, str):
        raise ValueError("TrainLens model_name must be a string or null")
    if framework is not None and not isinstance(framework, str):
        raise ValueError("TrainLens framework must be a string or null")
    return TrainingRun(
        run_id=run_id,
        created_at=created_at,
        model_name=model_name,
        framework=framework,
        metrics=metrics,
        parameters=parameters,
        metadata=metadata,
        notes=tuple(notes_raw),
    )


def _metric_from_dict(raw: Any) -> MetricSeries:
    if not isinstance(raw, dict):
        raise ValueError("TrainLens metric entries must be objects")
    name = raw.get("name")
    values = raw.get("values")
    steps = raw.get("steps", [])
    split = raw.get("split")
    if not isinstance(name, str) or not name:
        raise ValueError("TrainLens metric name must be a non-empty string")
    if not isinstance(values, list) or not all(_finite_number(item) for item in values):
        raise ValueError("TrainLens metric values must be finite numbers")
    if not isinstance(steps, list) or not all(
        item is None or _finite_number(item) for item in steps
    ):
        raise ValueError("TrainLens metric steps must be finite numbers or null")
    if split is not None and not isinstance(split, str):
        raise ValueError("TrainLens metric split must be a string or null")
    return MetricSeries(
        name=name,
        values=tuple(float(item) for item in values),
        split=split,
        steps=tuple(steps),
    )


def _simple_mapping(value: Any, label: str) -> dict[str, RunValue]:
    if not isinstance(value, dict):
        raise ValueError(f"TrainLens run {label} must be an object")
    output: dict[str, RunValue] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str | int | float | bool | type(None)):
            raise ValueError(f"TrainLens run {label} must contain JSON scalar values")
        if isinstance(item, float) and not isfinite(item):
            raise ValueError(f"TrainLens run {label} must contain finite numeric values")
        output[key] = item
    return output


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and isfinite(float(value))
    )
