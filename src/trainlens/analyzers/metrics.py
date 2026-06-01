"""Metric normalization from common training-history shapes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from trainlens.models.metric import MetricSeries

_TRAIN_PREFIXES = ("train_", "training_")
_VALIDATION_PREFIXES = ("val_", "valid_", "validation_", "eval_")
_ALIASES = {
    "loss/train": "train_loss",
    "loss/eval": "validation_loss",
    "eval/loss": "validation_loss",
    "train/loss": "train_loss",
    "eval_loss": "validation_loss",
    "train_loss": "train_loss",
    "train_losses": "train_loss",
    "training_losses": "train_loss",
    "val_loss": "validation_loss",
    "val_losses": "validation_loss",
    "valid_loss": "validation_loss",
    "valid_losses": "validation_loss",
    "validation_losses": "validation_loss",
    "perplexity": "perplexity",
    "eval_perplexity": "validation_perplexity",
    "validation_perplexity": "validation_perplexity",
    "clip_loss": "contrastive_loss",
    "image_text_loss": "contrastive_loss",
    "itc_loss": "contrastive_loss",
    "retrieval_recall": "retrieval_recall",
    "recall@1": "recall_at_1",
    "eval_recall@1": "validation_recall_at_1",
    "fad": "frechet_audio_distance",
    "frechet_audio_distance": "frechet_audio_distance",
    "eval_fad": "validation_frechet_audio_distance",
    "eval_frechet_audio_distance": "validation_frechet_audio_distance",
    "clap_score": "clap_score",
    "eval_clap_score": "validation_clap_score",
    "stft_loss": "stft_loss",
    "eval_stft_loss": "validation_stft_loss",
    "mel_loss": "mel_loss",
    "eval_mel_loss": "validation_mel_loss",
    "spectral_convergence": "spectral_convergence",
    "eval_spectral_convergence": "validation_spectral_convergence",
}


def extract_metric_series(namespace: Mapping[str, Any]) -> dict[str, MetricSeries]:
    candidates: dict[str, MetricSeries] = {}
    for name, value in namespace.items():
        if _looks_like_history_name(name):
            candidates.update(_series_from_mapping(value))
        candidates.update(_series_from_named_value(name, value))
    for key in (
        "history",
        "training_history",
        "metrics",
        "train_metrics",
        "validation_metrics",
        "val_metrics",
        "pytorch_metrics",
        "callback_metrics",
        "logged_metrics",
    ):
        candidates.update(_series_from_mapping(namespace.get(key)))
    return candidates


def paired_metric(
    series: dict[str, MetricSeries], base_name: str
) -> tuple[MetricSeries | None, MetricSeries | None]:
    train = series.get(f"train_{base_name}") or series.get(base_name)
    validation = series.get(f"validation_{base_name}") or series.get(f"val_{base_name}")
    return train, validation


def _looks_like_history_name(name: str) -> bool:
    lower = name.lower()
    return (
        "history" in lower
        or "log" in lower
        or lower in {"metrics", "losses", "accuracies", "callback_metrics", "logged_metrics"}
        or lower.endswith(("_loss", "_losses", "_accuracy", "_accuracies"))
    )


def _series_from_named_value(name: str, value: Any) -> dict[str, MetricSeries]:
    if _looks_like_log_history(value):
        return _series_from_log_history(value)
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return {}
    numeric = _coerce_floats(value)
    if not numeric:
        return {}
    normalized, split = _normalize_name(name)
    return {normalized: MetricSeries(name=normalized, values=tuple(numeric), split=split)}


def _series_from_mapping(value: Any) -> dict[str, MetricSeries]:
    if hasattr(value, "history"):
        value = value.history
    elif hasattr(value, "metrics"):
        value = value.metrics
    elif hasattr(value, "log_history"):
        value = value.log_history
    if _looks_like_log_history(value):
        return _series_from_log_history(value)
    if not isinstance(value, Mapping):
        return {}
    found: dict[str, MetricSeries] = {}
    for key, raw_values in value.items():
        if isinstance(raw_values, Sequence) and not isinstance(raw_values, str | bytes):
            numeric = _coerce_floats(raw_values)
            if numeric:
                normalized, split = _normalize_name(str(key))
                found[normalized] = MetricSeries(
                    name=normalized, values=tuple(numeric), split=split
                )
        else:
            numeric_value = _coerce_float(raw_values)
            if numeric_value is None:
                continue
            normalized, split = _normalize_name(str(key))
            found[normalized] = MetricSeries(
                name=normalized, values=(numeric_value,), split=split
            )
    return found


def _looks_like_log_history(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, str | bytes)
        and all(isinstance(item, Mapping) for item in value)
    )


def _series_from_log_history(value: Sequence[Any]) -> dict[str, MetricSeries]:
    grouped: dict[str, list[float]] = {}
    steps: dict[str, list[int]] = {}
    for index, entry in enumerate(value, start=1):
        if not isinstance(entry, Mapping):
            continue
        raw_step = _first_present(entry, "step", "global_step", "epoch")
        step = _coerce_step(index if raw_step is None else raw_step)
        for key, raw_value in entry.items():
            if key in {"step", "global_step", "epoch"}:
                continue
            numeric_value = _coerce_float(raw_value)
            if numeric_value is None:
                continue
            normalized, _split = _normalize_name(str(key))
            grouped.setdefault(normalized, []).append(numeric_value)
            if step is not None:
                steps.setdefault(normalized, []).append(step)
    return {
        name: MetricSeries(
            name=name,
            values=tuple(values),
            split=_split_from_name(name),
            steps=tuple(steps.get(name, ())),
        )
        for name, values in grouped.items()
    }


def _coerce_step(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_present(entry: Mapping[Any, Any], *names: str) -> Any:
    for name in names:
        if name in entry:
            return entry[name]
    return None


def _split_from_name(name: str) -> str | None:
    if name.startswith("train_"):
        return "train"
    if name.startswith("validation_"):
        return "validation"
    return None


def _coerce_floats(values: Sequence[Any]) -> list[float]:
    numeric: list[float] = []
    for value in values:
        numeric_value = _coerce_float(value)
        if numeric_value is None:
            return []
        numeric.append(numeric_value)
    return numeric


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (AttributeError, TypeError, ValueError):
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_name(name: str) -> tuple[str, str | None]:
    lower = name.lower().replace(" ", "_")
    if lower in _ALIASES:
        aliased = _ALIASES[lower]
        if aliased.startswith("train_"):
            return aliased, "train"
        if aliased.startswith("validation_"):
            return aliased, "validation"
        return aliased, None
    for prefix in _TRAIN_PREFIXES:
        if lower.startswith(prefix):
            return f"train_{lower.removeprefix(prefix)}", "train"
    for prefix in _VALIDATION_PREFIXES:
        if lower.startswith(prefix):
            return f"validation_{lower.removeprefix(prefix)}", "validation"
    return lower, None
