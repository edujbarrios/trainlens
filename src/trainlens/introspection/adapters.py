"""Framework-specific notebook evidence adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, cast, runtime_checkable

from trainlens.models.snapshot import FrameworkArtifact


class FrameworkAdapter(Protocol):
    """Extract training evidence from one framework-specific object shape."""

    name: str

    def can_handle(self, value: object) -> bool:
        """Return whether this adapter can safely inspect the value."""

    def extract(self, variable_name: str, value: object) -> FrameworkArtifact | None:
        """Extract normalized training evidence from the value."""


@runtime_checkable
class ItemScalar(Protocol):
    """Scalar tensor-like object that can return a Python value."""

    def item(self) -> object:
        """Return the scalar value."""


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


class PyTorchModuleAdapter:
    """Extract parameter counts from a plain PyTorch module-like object."""

    name = "pytorch_module"

    def can_handle(self, value: object) -> bool:
        has_parameters = callable(getattr(value, "parameters", None))
        return has_parameters and (
            _module_name(value).startswith("torch.nn.")
            or (
                callable(getattr(value, "state_dict", None))
                and callable(getattr(value, "train", None))
            )
        )

    def extract(self, variable_name: str, value: object) -> FrameworkArtifact | None:
        counts = _model_parameter_counts(value)
        return FrameworkArtifact(
            variable_name=variable_name,
            framework="pytorch",
            type_name=value.__class__.__name__,
            history={},
            training_parameters=counts,
            model_name=value.__class__.__name__,
            model_ref=value,
            confidence=0.9,
            reasons=("has PyTorch module parameters",),
        )


class PyTorchOptimizerAdapter:
    """Extract hyperparameters from a plain PyTorch optimizer-like object."""

    name = "pytorch_optimizer"

    def can_handle(self, value: object) -> bool:
        groups = getattr(value, "param_groups", None)
        return _module_name(value).startswith("torch.optim.") and isinstance(groups, Sequence)

    def extract(self, variable_name: str, value: object) -> FrameworkArtifact | None:
        groups = getattr(value, "param_groups", None)
        if not isinstance(groups, Sequence):
            return None
        parameters: dict[str, Any] = {
            "optimizer": value.__class__.__name__,
            "parameter_groups": len(groups),
        }
        for index, group in enumerate(groups):
            if not isinstance(group, Mapping):
                continue
            for key, raw_value in group.items():
                if str(key) == "params":
                    continue
                normalized = _training_parameter_value(raw_value)
                if normalized is not None:
                    parameters[f"group_{index}.{key}"] = normalized
        defaults = getattr(value, "defaults", None)
        if isinstance(defaults, Mapping):
            for key, raw_value in defaults.items():
                normalized = _training_parameter_value(raw_value)
                if normalized is not None:
                    parameters.setdefault(f"default.{key}", normalized)
        return FrameworkArtifact(
            variable_name=variable_name,
            framework="pytorch",
            type_name=value.__class__.__name__,
            history={},
            training_parameters=parameters,
            confidence=0.92,
            reasons=("has PyTorch optimizer parameter groups",),
        )


class PyTorchSchedulerAdapter:
    """Extract state from a plain PyTorch learning-rate scheduler-like object."""

    name = "pytorch_scheduler"

    def can_handle(self, value: object) -> bool:
        return _module_name(value).startswith("torch.optim.lr_scheduler") and (
            hasattr(value, "last_epoch") or callable(getattr(value, "get_last_lr", None))
        )

    def extract(self, variable_name: str, value: object) -> FrameworkArtifact | None:
        parameters: dict[str, Any] = {"scheduler": value.__class__.__name__}
        state_dict = getattr(value, "state_dict", None)
        if callable(state_dict):
            try:
                state = state_dict()
            except (AttributeError, RuntimeError, TypeError, ValueError):
                state = None
            if isinstance(state, Mapping):
                for key, raw_value in state.items():
                    normalized = _training_parameter_value(raw_value)
                    if normalized is not None:
                        parameters[str(key)] = normalized
        last_epoch = _coerce_int(getattr(value, "last_epoch", None))
        if last_epoch is not None:
            parameters["last_epoch"] = last_epoch
        get_last_lr = getattr(value, "get_last_lr", None)
        if callable(get_last_lr):
            try:
                last_lr = _training_parameter_value(get_last_lr())
            except (AttributeError, RuntimeError, TypeError, ValueError):
                last_lr = None
            if last_lr is not None:
                parameters["last_lr"] = last_lr
        return FrameworkArtifact(
            variable_name=variable_name,
            framework="pytorch",
            type_name=value.__class__.__name__,
            history={},
            training_parameters=parameters,
            confidence=0.88,
            reasons=("has PyTorch scheduler state",),
        )


class PyTorchDataLoaderAdapter:
    """Extract batching parameters from a plain PyTorch DataLoader-like object."""

    name = "pytorch_dataloader"

    def can_handle(self, value: object) -> bool:
        return _module_name(value).startswith("torch.utils.data.") and hasattr(
            value, "dataset"
        )

    def extract(self, variable_name: str, value: object) -> FrameworkArtifact | None:
        parameters: dict[str, Any] = {"loader": value.__class__.__name__}
        for name in (
            "batch_size",
            "num_workers",
            "drop_last",
            "pin_memory",
            "persistent_workers",
            "prefetch_factor",
            "timeout",
        ):
            normalized = _training_parameter_value(getattr(value, name, None))
            if normalized is not None:
                parameters[name] = normalized
        dataset_size = _safe_len(getattr(value, "dataset", None))
        if dataset_size is not None:
            parameters["dataset_size"] = dataset_size
        return FrameworkArtifact(
            variable_name=variable_name,
            framework="pytorch",
            type_name=value.__class__.__name__,
            history={},
            training_parameters=parameters,
            confidence=0.88,
            reasons=("has PyTorch data-loader settings",),
        )


DEFAULT_ADAPTERS: tuple[FrameworkAdapter, ...] = (
    KerasHistoryAdapter(),
    HuggingFaceTrainerAdapter(),
    LightningTrainerAdapter(),
    PyTorchModuleAdapter(),
    PyTorchOptimizerAdapter(),
    PyTorchSchedulerAdapter(),
    PyTorchDataLoaderAdapter(),
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
    log_entries = cast(Sequence[Mapping[Any, Any]], value)
    entries: list[dict[str, float | int]] = []
    for entry in log_entries:
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
    if isinstance(value, ItemScalar):
        try:
            value = value.item()
        except (AttributeError, TypeError, ValueError):
            return None
    if isinstance(value, int | float | str | bytes | bytearray):
        try:
            return float(value)
        except ValueError:
            return None
    try:
        return float(cast(Any, value))
    except (TypeError, ValueError):
        return None


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, ItemScalar):
        try:
            value = value.item()
        except (AttributeError, TypeError, ValueError):
            return None
    if isinstance(value, int):
        return value
    if isinstance(value, float | str | bytes | bytearray):
        try:
            return int(value)
        except ValueError:
            return None
    try:
        return int(cast(Any, value))
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


def _training_parameter_value(value: object) -> Any | None:
    if isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        normalized = tuple(_training_parameter_value(item) for item in value)
        if all(item is not None for item in normalized):
            return normalized
    return None


def _model_parameter_counts(value: object) -> dict[str, Any]:
    parameters = getattr(value, "parameters", None)
    if not callable(parameters):
        return {}
    total = 0
    trainable = 0
    try:
        for parameter in parameters():
            numel = _coerce_int(getattr(parameter, "numel", lambda: None)())
            if numel is None:
                continue
            total += numel
            if bool(getattr(parameter, "requires_grad", False)):
                trainable += numel
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return {}
    result: dict[str, Any] = {
        "total_parameters": total,
        "trainable_parameters": trainable,
    }
    if total:
        result["trainable_ratio"] = trainable / total
    return result


def _safe_len(value: object) -> int | None:
    try:
        return len(value)  # type: ignore[arg-type]
    except (RuntimeError, TypeError, ValueError):
        return None
