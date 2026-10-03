"""Metric normalization from common training-history shapes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any, Literal

from trainlens.metric_semantics import metric_direction
from trainlens.models.metric import MetricSeries

MetricSplit = Literal["train", "validation", "test"]

_TRAIN_PREFIXES = ("train_", "training_")
_VALIDATION_PREFIXES = ("val_", "valid_", "validation_", "eval_")
_TEST_PREFIXES = ("test_", "testing_", "holdout_")
_MAX_ARRAY_LIKE_POINTS = 100_000
_MISSING = object()
_ALIASES = {
    "loss/train": "train_loss",
    "loss/eval": "validation_loss",
    "loss/val": "validation_loss",
    "loss/validation": "validation_loss",
    "loss/test": "test_loss",
    "eval/loss": "validation_loss",
    "validation/loss": "validation_loss",
    "test/loss": "test_loss",
    "holdout/loss": "test_loss",
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
    "test_losses": "test_loss",
    "testing_loss": "test_loss",
    "testing_losses": "test_loss",
    "holdout_loss": "test_loss",
    "holdout_losses": "test_loss",
    "perplexity": "perplexity",
    "eval_perplexity": "validation_perplexity",
    "validation_perplexity": "validation_perplexity",
    "test_perplexity": "test_perplexity",
    "clip_loss": "contrastive_loss",
    "image_text_loss": "contrastive_loss",
    "itc_loss": "contrastive_loss",
    "retrieval_recall": "retrieval_recall",
    "recall@1": "recall_at_1",
    "eval_recall@1": "validation_recall_at_1",
    "test_recall@1": "test_recall_at_1",
    "fad": "frechet_audio_distance",
    "frechet_audio_distance": "frechet_audio_distance",
    "eval_fad": "validation_frechet_audio_distance",
    "eval_frechet_audio_distance": "validation_frechet_audio_distance",
    "test_fad": "test_frechet_audio_distance",
    "test_frechet_audio_distance": "test_frechet_audio_distance",
    "clap_score": "clap_score",
    "eval_clap_score": "validation_clap_score",
    "test_clap_score": "test_clap_score",
    "stft_loss": "stft_loss",
    "eval_stft_loss": "validation_stft_loss",
    "test_stft_loss": "test_stft_loss",
    "mel_loss": "mel_loss",
    "eval_mel_loss": "validation_mel_loss",
    "test_mel_loss": "test_mel_loss",
    "spectral_convergence": "spectral_convergence",
    "eval_spectral_convergence": "validation_spectral_convergence",
    "test_spectral_convergence": "test_spectral_convergence",
}


def extract_metric_series(namespace: Mapping[str, Any]) -> dict[str, MetricSeries]:
    """Extract normalized metric histories and scalar results from a namespace."""

    candidates: dict[str, MetricSeries] = {}
    for name, value in namespace.items():
        if _looks_like_history_name(name):
            candidates.update(
                _series_from_mapping(value, default_split=_container_split(name))
            )
        candidates.update(_series_from_named_value(name, value))
    for key, split in (
        ("history", None),
        ("training_history", "train"),
        ("metrics", None),
        ("train_metrics", "train"),
        ("validation_metrics", "validation"),
        ("val_metrics", "validation"),
        ("eval_metrics", "validation"),
        ("test_metrics", "test"),
        ("testing_metrics", "test"),
        ("holdout_metrics", "test"),
        ("pytorch_metrics", None),
        ("callback_metrics", None),
        ("logged_metrics", None),
    ):
        candidates.update(_series_from_mapping(namespace.get(key), default_split=split))
    return candidates


def metric_splits(
    series: Mapping[str, MetricSeries], base_name: str
) -> dict[MetricSplit, MetricSeries]:
    """Return train/validation/test series for one normalized metric family.

    Unprefixed histories such as Keras ``accuracy`` or Hugging Face ``loss`` are
    treated as training observations when a split-aware view is requested.
    """

    normalized = _base_metric_name(base_name)
    found: dict[MetricSplit, MetricSeries] = {}
    train = series.get(f"train_{normalized}") or series.get(normalized)
    validation = series.get(f"validation_{normalized}") or series.get(f"val_{normalized}")
    test = series.get(f"test_{normalized}")
    if train is not None:
        found["train"] = train
    if validation is not None:
        found["validation"] = validation
    if test is not None:
        found["test"] = test
    return found


def paired_metric(
    series: Mapping[str, MetricSeries], base_name: str
) -> tuple[MetricSeries | None, MetricSeries | None]:
    """Backward-compatible train/validation lookup for one metric family."""

    splits = metric_splits(series, base_name)
    return splits.get("train"), splits.get("validation")


def _looks_like_history_name(name: str) -> bool:
    lower = name.lower()
    return (
        "history" in lower
        or "log" in lower
        or lower
        in {
            "metrics",
            "losses",
            "accuracies",
            "callback_metrics",
            "logged_metrics",
            "test_metrics",
        }
        or lower.endswith("_metrics")
        or lower.endswith(("_loss", "_losses", "_accuracy", "_accuracies"))
    )


def _container_split(name: str) -> MetricSplit | None:
    normalized = name.lower().replace("-", "_").replace("/", "_")
    if normalized.startswith(_TRAIN_PREFIXES) or normalized == "training_history":
        return "train"
    if normalized.startswith(_VALIDATION_PREFIXES):
        return "validation"
    if normalized.startswith(_TEST_PREFIXES):
        return "test"
    return None


def _series_from_named_value(name: str, value: Any) -> dict[str, MetricSeries]:
    if _looks_like_log_history(value):
        return _series_from_log_history(value)
    normalized, split = _normalize_name(name)
    values = _history_values(value)
    if values is not None:
        numeric = _coerce_floats(values)
        if not numeric:
            return {}
        return {normalized: MetricSeries(name=normalized, values=tuple(numeric), split=split)}
    if not _looks_like_scalar_metric_name(name):
        return {}
    numeric_value = _coerce_float(value)
    if numeric_value is None:
        return {}
    return {
        normalized: MetricSeries(name=normalized, values=(numeric_value,), split=split)
    }


def _looks_like_scalar_metric_name(name: str) -> bool:
    lower = name.lower().replace(" ", "_")
    normalized = lower.replace("/", "_").replace("-", "_")
    base = _base_metric_name(normalized)
    has_split_prefix = normalized.startswith(
        (*_TRAIN_PREFIXES, *_VALIDATION_PREFIXES, *_TEST_PREFIXES)
    )
    return lower in _ALIASES or has_split_prefix or metric_direction(base) is not None


def _series_from_mapping(
    value: Any, *, default_split: MetricSplit | None = None
) -> dict[str, MetricSeries]:
    for attribute in ("history", "metrics", "log_history"):
        nested = _safe_getattr(value, attribute, _MISSING)
        if nested is not _MISSING:
            value = nested
            break
    if _looks_like_log_history(value):
        return _series_from_log_history(value)
    if not isinstance(value, Mapping):
        return {}
    found: dict[str, MetricSeries] = {}
    for key, raw_values in value.items():
        normalized, split = _normalize_name(str(key))
        if split is None and default_split is not None:
            normalized = f"{default_split}_{normalized}"
            split = default_split
        history_values = _history_values(raw_values)
        if history_values is not None:
            numeric = _coerce_floats(history_values)
            if numeric:
                found[normalized] = MetricSeries(
                    name=normalized, values=tuple(numeric), split=split
                )
        else:
            numeric_value = _coerce_float(raw_values)
            if numeric_value is None:
                continue
            found[normalized] = MetricSeries(
                name=normalized, values=(numeric_value,), split=split
            )
    return found


def _history_values(value: Any) -> tuple[Any, ...] | None:
    """Return a bounded 1-D history without consuming arbitrary iterables."""

    if isinstance(value, str | bytes | bytearray | Mapping):
        return None
    if isinstance(value, Sequence):
        try:
            return tuple(value)
        except (IndexError, RuntimeError, TypeError, ValueError):
            return None

    shape = _safe_getattr(value, "shape", _MISSING)
    if shape is _MISSING or shape is None:
        return None
    try:
        dimensions = tuple(shape)
    except (RuntimeError, TypeError, ValueError):
        return None
    if len(dimensions) != 1:
        return None
    try:
        length = len(value)
    except (RuntimeError, TypeError, ValueError):
        return None
    if length > _MAX_ARRAY_LIKE_POINTS:
        return None
    try:
        return tuple(value)
    except (IndexError, RuntimeError, TypeError, ValueError):
        return None


def _looks_like_log_history(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, str | bytes)
        and all(isinstance(item, Mapping) for item in value)
    )


def _series_from_log_history(value: Sequence[Any]) -> dict[str, MetricSeries]:
    grouped: dict[str, list[float]] = {}
    steps: dict[str, list[int | float | None]] = {}
    for index, entry in enumerate(value, start=1):
        if not isinstance(entry, Mapping):
            continue
        step = _observation_step(entry, index)
        for key, raw_value in entry.items():
            if key in {"step", "global_step", "epoch"}:
                continue
            numeric_value = _coerce_float(raw_value)
            if numeric_value is None:
                continue
            normalized, _split = _normalize_name(str(key))
            grouped.setdefault(normalized, []).append(numeric_value)
            steps.setdefault(normalized, []).append(step)
    return {
        name: MetricSeries(
            name=name,
            values=tuple(values),
            split=_split_from_name(name),
            steps=tuple(steps[name]),
        )
        for name, values in grouped.items()
    }


def _observation_step(entry: Mapping[Any, Any], default: int) -> int | float | None:
    saw_step_metadata = False
    for name in ("step", "global_step"):
        if name not in entry:
            continue
        saw_step_metadata = True
        step = _coerce_integer_step(entry[name])
        if step is not None:
            return step
    if "epoch" in entry:
        saw_step_metadata = True
        epoch = _coerce_float(entry["epoch"])
        if epoch is not None:
            return epoch
    return None if saw_step_metadata else default


def _coerce_integer_step(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    item = _safe_getattr(value, "item", _MISSING)
    if item is not _MISSING:
        try:
            value = item()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if isfinite(value) and value.is_integer() else None
    try:
        return int(value)
    except (OverflowError, TypeError, ValueError):
        return None


def _split_from_name(name: str) -> str | None:
    if name.startswith("train_"):
        return "train"
    if name.startswith("validation_"):
        return "validation"
    if name.startswith("test_"):
        return "test"
    return None


def _coerce_floats(values: Sequence[Any]) -> list[float]:
    """Preserve finite observations and omit invalid/non-finite points."""

    numeric: list[float] = []
    for value in values:
        numeric_value = _coerce_float(value)
        if numeric_value is not None:
            numeric.append(numeric_value)
    return numeric


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    item = _safe_getattr(value, "item", _MISSING)
    if item is not _MISSING:
        try:
            value = item()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    return numeric_value if isfinite(numeric_value) else None


def _safe_getattr(value: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(value, name, default)
    except Exception:
        return default


def _base_metric_name(name: str) -> str:
    normalized = name.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
    for prefix in (*_TRAIN_PREFIXES, *_VALIDATION_PREFIXES, *_TEST_PREFIXES):
        if normalized.startswith(prefix):
            return normalized.removeprefix(prefix)
    return normalized


def _normalize_name(name: str) -> tuple[str, str | None]:
    lower = name.lower().replace(" ", "_")
    if lower in _ALIASES:
        aliased = _ALIASES[lower]
        return aliased, _split_from_name(aliased)
    lower = lower.replace("/", "_").replace("-", "_")
    for prefix in _TRAIN_PREFIXES:
        if lower.startswith(prefix):
            return f"train_{lower.removeprefix(prefix)}", "train"
    for prefix in _VALIDATION_PREFIXES:
        if lower.startswith(prefix):
            return f"validation_{lower.removeprefix(prefix)}", "validation"
    for prefix in _TEST_PREFIXES:
        if lower.startswith(prefix):
            return f"test_{lower.removeprefix(prefix)}", "test"
    return lower, None
