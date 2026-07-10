"""Framework-specific notebook evidence adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from trainlens.models.snapshot import FrameworkArtifact


class FrameworkAdapter(Protocol):
    """Extract training evidence from one framework-specific object shape."""

    name: str

    def can_handle(self, value: object) -> bool:
        """Return whether this adapter can safely inspect the value."""

    def extract(self, variable_name: str, value: object) -> FrameworkArtifact | None:
        """Extract normalized training evidence from the value."""


class KerasHistoryAdapter:
    """Extract metrics from Keras History-like objects."""

    name = "keras"

    def can_handle(self, value: object) -> bool:
        history = getattr(value, "history", None)
        if not isinstance(history, Mapping):
            return False
        type_name = value.__class__.__name__.lower()
        module = _module_name(value)
        return (
            type_name == "history"
            or module.startswith(("keras.", "tensorflow."))
            or any(_looks_like_metric_key(str(key)) for key in history)
        )

    def extract(self, variable_name: str, value: object) -> FrameworkArtifact | None:
        history = _history_mapping(getattr(value, "history", None))
        if not history:
            return None
        return FrameworkArtifact(
            variable_name=variable_name,
            framework="keras",
            type_name=value.__class__.__name__,
            history=history,
            confidence=0.86,
            reasons=("has Keras-style history mapping",),
        )


class HuggingFaceTrainerAdapter:
    """Extract metrics and model metadata from Hugging Face Trainer-like objects."""

    name = "huggingface"

    def can_handle(self, value: object) -> bool:
        module = _module_name(value)
        state = getattr(value, "state", None)
        log_history = getattr(state, "log_history", None) or getattr(value, "log_history", None)
        return (
            module.startswith("transformers.")
            or (
                _looks_like_log_history(log_history)
                and (
                    hasattr(value, "args")
                    or hasattr(value, "model")
                    or callable(getattr(value, "evaluate", None))
                )
            )
        )

    def extract(self, variable_name: str, value: object) -> FrameworkArtifact | None:
        state = getattr(value, "state", None)
        log_history = _log_history(
            getattr(state, "log_history", None) or getattr(value, "log_history", None)
        )
        if not log_history:
            return None
        model_ref = getattr(value, "model", None)
        metadata = _trainer_args_metadata(getattr(value, "args", None))
        return FrameworkArtifact(
            variable_name=variable_name,
            framework="huggingface",
            type_name=value.__class__.__name__,
            history={},
            log_history=log_history,
            latest_metrics=_latest_metrics(log_history),
            model_name=_display_name(model_ref),
            model_ref=model_ref,
            confidence=0.9,
            reasons=("has Trainer log history", *metadata),
        )


class LightningTrainerAdapter:
    """Extract metrics from PyTorch Lightning Trainer-like objects."""

    name = "lightning"

    def can_handle(self, value: object) -> bool:
        module = _module_name(value)
        metrics = _first_metric_mapping(
            value,
            "callback_metrics",
            "logged_metrics",
            "progress_bar_metrics",
        )
        return (
            module.startswith(("lightning.", "pytorch_lightning."))
            or (
                metrics is not None
                and (hasattr(value, "current_epoch") or hasattr(value, "global_step"))
            )
        )

    def extract(self, variable_name: str, value: object) -> FrameworkArtifact | None:
        metrics = _first_metric_mapping(
            value,
            "callback_metrics",
            "logged_metrics",
            "progress_bar_metrics",
        )
        latest_metrics = _metric_mapping(metrics)
        if not latest_metrics:
            return None
        model_ref = (
            getattr(value, "lightning_module", None)
            or getattr(value, "model", None)
            or getattr(value, "module", None)
        )
        return FrameworkArtifact(
            variable_name=variable_name,
            framework="lightning",
            type_name=value.__class__.__name__,
            history={},
            latest_metrics=latest_metrics,
            model_name=_display_name(model_ref),
            model_ref=model_ref,
            confidence=0.84,
            reasons=("has Lightning metric mappings",),
        )


DEFAULT_ADAPTERS: tuple[FrameworkAdapter, ...] = (
    KerasHistoryAdapter(),
    HuggingFaceTrainerAdapter(),
    LightningTrainerAdapter(),
)


def extract_framework_artifact(
    variable_name: str,
    value: object,
    adapters: Sequence[FrameworkAdapter] = DEFAULT_ADAPTERS,
) -> FrameworkArtifact | None:
    """Return framework evidence from the first adapter that recognizes value."""

    for adapter in adapters:
        if not adapter.can_handle(value):
            continue
        return adapter.extract(variable_name, value)
    return None


def _module_name(value: object) -> str:
    return getattr(value.__class__, "__module__", "") or ""


def _display_name(value: object | None) -> str | None:
    if value is None:
        return None
    return value.__class__.__name__


def _history_mapping(value: object) -> dict[str, tuple[float, ...]]:
    if not isinstance(value, Mapping):
        return {}
    history: dict[str, tuple[float, ...]] = {}
    for key, raw_values in value.items():
        if isinstance(raw_values, Sequence) and not isinstance(raw_values, str | bytes):
            values = _coerce_float_tuple(raw_values)
        else:
            scalar = _coerce_float(raw_values)
            values = () if scalar is None else (scalar,)
        if values:
            history[str(key)] = values
    return history


def _log_history(value: object) -> tuple[dict[str, float | int], ...]:
    if not _looks_like_log_history(value):
        return ()
    entries: list[dict[str, float | int]] = []
    for entry in value:
        if not isinstance(entry, Mapping):
            continue
        normalized: dict[str, float | int] = {}
        for key, raw_value in entry.items():
            if str(key) in {"step", "global_step", "epoch"}:
                step = _coerce_int(raw_value)
                if step is not None:
                    normalized[str(key)] = step
                continue
            numeric = _coerce_float(raw_value)
            if numeric is not None:
                normalized[str(key)] = numeric
        if normalized:
            entries.append(normalized)
    return tuple(entries)


def _looks_like_log_history(value: object) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, str | bytes)
        and all(isinstance(item, Mapping) for item in value)
    )


def _latest_metrics(log_history: Sequence[Mapping[str, float | int]]) -> dict[str, float]:
    latest: dict[str, float] = {}
    for entry in log_history:
        for key, value in entry.items():
            if key in {"step", "global_step", "epoch"}:
                continue
            latest[key] = float(value)
    return latest


def _first_metric_mapping(value: object, *names: str) -> Mapping[Any, Any] | None:
    for name in names:
        metrics = getattr(value, name, None)
        if isinstance(metrics, Mapping):
            return metrics
    return None


def _metric_mapping(value: Mapping[Any, Any] | None) -> dict[str, float]:
    if value is None:
        return {}
    found: dict[str, float] = {}
    for key, raw_value in value.items():
        numeric = _coerce_float(raw_value)
        if numeric is not None:
            found[str(key)] = numeric
    return found


def _coerce_float_tuple(values: Sequence[object]) -> tuple[float, ...]:
    numeric: list[float] = []
    for value in values:
        numeric_value = _coerce_float(value)
        if numeric_value is None:
            return ()
        numeric.append(numeric_value)
    return tuple(numeric)


def _coerce_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()  # type: ignore[assignment]
        except (AttributeError, TypeError, ValueError):
            return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _coerce_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _looks_like_metric_key(key: str) -> bool:
    lower = key.lower()
    return any(token in lower for token in ("loss", "accuracy", "acc", "metric", "eval", "val"))


def _trainer_args_metadata(args: object | None) -> tuple[str, ...]:
    if args is None:
        return ()
    names = ("learning_rate", "num_train_epochs", "per_device_train_batch_size")
    found = tuple(name for name in names if getattr(args, name, None) is not None)
    if not found:
        return ()
    return (f"has Trainer args: {', '.join(found)}",)
