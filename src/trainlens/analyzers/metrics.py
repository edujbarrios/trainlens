"""Metric normalization from common training-history shapes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from trainlens.models.metric import MetricSeries

_TRAIN_PREFIXES = ("train_", "training_")
_VALIDATION_PREFIXES = ("val_", "valid_", "validation_", "eval_")


def extract_metric_series(namespace: Mapping[str, Any]) -> dict[str, MetricSeries]:
    candidates: dict[str, MetricSeries] = {}
    for name, value in namespace.items():
        if _looks_like_history_name(name):
            candidates.update(_series_from_mapping(value))
    for key in ("history", "training_history", "metrics"):
        candidates.update(_series_from_mapping(namespace.get(key)))
    return candidates


def paired_metric(series: dict[str, MetricSeries], base_name: str) -> tuple[MetricSeries | None, MetricSeries | None]:
    train = series.get(f"train_{base_name}") or series.get(base_name)
    validation = series.get(f"validation_{base_name}") or series.get(f"val_{base_name}")
    return train, validation


def _looks_like_history_name(name: str) -> bool:
    lower = name.lower()
    return "history" in lower or lower in {"metrics", "logs"}


def _series_from_mapping(value: Any) -> dict[str, MetricSeries]:
    if hasattr(value, "history"):
        value = getattr(value, "history")
    if not isinstance(value, Mapping):
        return {}
    found: dict[str, MetricSeries] = {}
    for key, raw_values in value.items():
        if isinstance(raw_values, Sequence) and not isinstance(raw_values, (str, bytes)):
            numeric = _coerce_floats(raw_values)
            if numeric:
                normalized, split = _normalize_name(str(key))
                found[normalized] = MetricSeries(name=normalized, values=tuple(numeric), split=split)
        elif isinstance(raw_values, (int, float)):
            normalized, split = _normalize_name(str(key))
            found[normalized] = MetricSeries(name=normalized, values=(float(raw_values),), split=split)
    return found


def _coerce_floats(values: Sequence[Any]) -> list[float]:
    numeric: list[float] = []
    for value in values:
        try:
            numeric.append(float(value))
        except (TypeError, ValueError):
            return []
    return numeric


def _normalize_name(name: str) -> tuple[str, str | None]:
    lower = name.lower()
    for prefix in _TRAIN_PREFIXES:
        if lower.startswith(prefix):
            return f"train_{lower.removeprefix(prefix)}", "train"
    for prefix in _VALIDATION_PREFIXES:
        if lower.startswith(prefix):
            return f"validation_{lower.removeprefix(prefix)}", "validation"
    return lower, None
