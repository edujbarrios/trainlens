"""Execution trace extraction from common notebook training artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from trainlens.models.trace import TraceEvent

_TRACE_NAMES = {
    "trace",
    "traces",
    "execution_trace",
    "execution_traces",
    "training_trace",
    "training_traces",
    "epoch_logs",
    "log_history",
    "logs",
    "train_logs",
    "validation_logs",
    "val_logs",
}
_META_KEYS = {"step", "global_step", "epoch", "event", "name", "message", "msg"}


def extract_trace_events(namespace: Mapping[str, Any], max_events: int = 8) -> list[TraceEvent]:
    """Return recent execution events from notebook variables and trainer state."""

    events: list[TraceEvent] = []
    for name, value in namespace.items():
        if _looks_like_trace_name(name):
            events.extend(_events_from_value(value))

    trainer_state = namespace.get("trainer_state")
    if trainer_state is not None:
        events.extend(_events_from_value(getattr(trainer_state, "log_history", None)))

    trainer = namespace.get("trainer")
    if trainer is not None:
        state = getattr(trainer, "state", None)
        events.extend(_events_from_value(getattr(state, "log_history", None)))

    return events[-max_events:]


def _events_from_value(value: Any) -> list[TraceEvent]:
    if hasattr(value, "log_history"):
        value = value.log_history
    if isinstance(value, Mapping):
        return [_event_from_mapping(value)]
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        events: list[TraceEvent] = []
        for item in value:
            if isinstance(item, Mapping):
                events.append(_event_from_mapping(item))
            elif isinstance(item, str):
                events.append(TraceEvent(message=item))
        return events
    return []


def _event_from_mapping(entry: Mapping[Any, Any]) -> TraceEvent:
    metrics: dict[str, float] = {}
    for key, value in entry.items():
        if str(key) in _META_KEYS:
            continue
        numeric_value = _coerce_float(value)
        if numeric_value is not None:
            metrics[str(key)] = numeric_value
    return TraceEvent(
        step=_coerce_int(_first_present(entry, "step", "global_step")),
        epoch=_coerce_float(entry.get("epoch")),
        name=_coerce_str(_first_present(entry, "event", "name")),
        message=_coerce_str(_first_present(entry, "message", "msg")),
        metrics=metrics,
    )


def _first_present(entry: Mapping[Any, Any], *names: str) -> Any:
    for name in names:
        if name in entry:
            return entry[name]
    return None


def _looks_like_trace_name(name: str) -> bool:
    lower = name.lower()
    return lower in _TRACE_NAMES or "trace" in lower or lower.endswith("_logs")


def _coerce_int(value: Any) -> int | None:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (AttributeError, TypeError, ValueError):
            return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
