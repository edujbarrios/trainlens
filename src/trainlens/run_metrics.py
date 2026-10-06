"""Split-aware helpers for portable training-run metrics."""

from __future__ import annotations

import re
from math import isfinite

from trainlens.models.metric import MetricSeries
from trainlens.models.run import TrainingRun

_SPLIT_PREFIXES = {
    "train": ("train", "training"),
    "validation": ("validation", "val", "valid", "eval"),
    "test": ("test", "testing", "holdout", "heldout", "held_out"),
}


def qualified_metric_name(series: MetricSeries) -> str:
    """Return a split-aware metric key without silently discarding split metadata."""

    name = series.name.strip()
    if not name:
        raise ValueError("metric series name must be non-empty")
    split = _normalize_split(series.split)
    if split is None:
        return name

    normalized = _normalize_name(name)
    prefixes = _SPLIT_PREFIXES.get(split, (split,))
    if any(
        normalized == prefix or normalized.startswith(f"{prefix}_")
        for prefix in prefixes
    ):
        return name
    return f"{split}_{name}"


def metric_series_by_name(run: TrainingRun) -> dict[str, MetricSeries]:
    """Index run series by split-aware names and reject ambiguous collisions."""

    output: dict[str, MetricSeries] = {}
    for series in run.metrics:
        key = qualified_metric_name(series)
        if key in output:
            raise ValueError(
                f"training run contains multiple metric series resolving to {key!r}; "
                "use unique metric names/splits"
            )
        output[key] = series
    return output


def final_metric_values(run: TrainingRun) -> dict[str, float]:
    """Return finite final metrics keyed by their split-aware names."""

    output: dict[str, float] = {}
    for name, series in metric_series_by_name(run).items():
        value = series.last
        if value is None or isinstance(value, bool) or not isfinite(value):
            continue
        output[name] = float(value)
    return output


def resolve_metric_series(run: TrainingRun, metric: str) -> MetricSeries:
    """Resolve one metric without choosing silently between train/validation/test."""

    indexed = metric_series_by_name(run)
    if metric in indexed:
        return indexed[metric]

    raw_matches = [series for series in run.metrics if series.name == metric]
    if len(raw_matches) == 1:
        return raw_matches[0]
    if len(raw_matches) > 1:
        choices = ", ".join(sorted(qualified_metric_name(series) for series in raw_matches))
        raise ValueError(
            f"metric {metric!r} is ambiguous across splits; choose one of: {choices}"
        )
    raise KeyError(metric)


def metric_namespace_from_run(run: TrainingRun) -> dict[str, object]:
    """Build analyzer-compatible histories while preserving split names and steps."""

    metric_series_by_name(run)
    namespace: dict[str, object] = {}
    for series_index, series in enumerate(run.metrics, start=1):
        if not series.values:
            continue
        name = qualified_metric_name(series)
        history: list[dict[str, int | float | None]] = []
        for index, value in enumerate(series.values):
            entry: dict[str, int | float | None] = {name: value}
            if series.steps:
                step = series.steps[index]
                if isinstance(step, float) and not step.is_integer():
                    entry["epoch"] = step
                else:
                    entry["step"] = step
            history.append(entry)
        safe_name = re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_") or "metric"
        namespace[f"trainlens_run_{series_index}_{safe_name}_log_history"] = history
    return namespace


def _normalize_split(split: str | None) -> str | None:
    if split is None:
        return None
    normalized = _normalize_name(split)
    if normalized in {"train", "training"}:
        return "train"
    if normalized in {"validation", "val", "valid", "eval"}:
        return "validation"
    if normalized in {"test", "testing", "holdout", "heldout", "held_out"}:
        return "test"
    return normalized


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
