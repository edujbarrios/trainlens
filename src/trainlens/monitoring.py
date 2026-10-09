"""Framework-neutral, real-time monitoring for training metrics."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

AlertSeverity = Literal["info", "warning", "critical"]


@dataclass(frozen=True)
class TrainingObservation:
    """A normalized snapshot received during training."""

    step: int
    metrics: Mapping[str, float]


@dataclass(frozen=True)
class TrainingAlert:
    """An evidence-backed issue detected in a metric stream."""

    code: str
    severity: AlertSeverity
    message: str
    evidence: tuple[str, ...]
    step: int
    persistent: bool = False


@dataclass(frozen=True)
class MonitorConfig:
    """Thresholds controlling deterministic live-training alerts."""

    patience: int = 3
    min_delta: float = 0.0
    detect_non_finite: bool = True
    detect_stagnation: bool = True
    detect_overfitting: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.patience, bool) or not isinstance(self.patience, int):
            raise TypeError("patience must be an integer")
        if self.patience < 2:
            raise ValueError("patience must be at least 2")
        if isinstance(self.min_delta, bool) or not isinstance(self.min_delta, int | float):
            raise TypeError("min_delta must be a finite number")
        if not math.isfinite(self.min_delta):
            raise ValueError("min_delta must be finite")
        if self.min_delta < 0:
            raise ValueError("min_delta cannot be negative")


AlertHandler = Callable[[TrainingAlert], None]
AlertDetector = Callable[
    [tuple[TrainingObservation, ...], TrainingObservation, MonitorConfig],
    Iterable[TrainingAlert],
]


class TrainLensMonitor:
    """Observe metric updates and emit deterministic alerts as training runs."""

    def __init__(
        self,
        config: MonitorConfig | None = None,
        *,
        on_alert: AlertHandler | None = None,
        detectors: Iterable[AlertDetector] = (),
    ) -> None:
        self.config = config or MonitorConfig()
        self._on_alert = on_alert
        self._detectors = list(detectors)
        self._observations: list[TrainingObservation] = []
        self._emitted_events: set[tuple[str, int, tuple[str, ...]]] = set()
        self._active_persistent: set[str] = set()

    @property
    def observations(self) -> tuple[TrainingObservation, ...]:
        """Return immutable access to observations received so far."""

        return tuple(self._observations)

    @property
    def detectors(self) -> tuple[AlertDetector, ...]:
        """Return the custom alert detectors registered on this monitor."""

        return tuple(self._detectors)

    def add_detector(self, detector: AlertDetector) -> None:
        """Register a custom deterministic alert detector for future observations."""

        self._detectors.append(detector)

    def observe(self, step: int, metrics: Mapping[str, float | int]) -> tuple[TrainingAlert, ...]:
        """Record one metric snapshot and return alerts triggered by it.

        Repeated updates for the same step are merged into one historical observation so
        framework callbacks cannot advance patience windows without training advancing.
        """

        if isinstance(step, bool) or not isinstance(step, int):
            raise TypeError("step must be an integer")
        if step < 0:
            raise ValueError("step cannot be negative")
        if self._observations and step < self._observations[-1].step:
            raise ValueError("step cannot move backwards")
        normalized: dict[str, float] = {}
        for name, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"metric {name!r} must be a number")
            normalized[str(name)] = float(value)

        if self._observations and step == self._observations[-1].step:
            merged = dict(self._observations[-1].metrics)
            merged.update(normalized)
            observation = TrainingObservation(step=step, metrics=MappingProxyType(merged))
            self._observations[-1] = observation
        else:
            observation = TrainingObservation(step=step, metrics=MappingProxyType(normalized))
            self._observations.append(observation)

        alerts = self._evaluate(observation, event_metrics=normalized)
        fresh = self._fresh_alerts(alerts)
        if self._on_alert is not None:
            for alert in fresh:
                self._on_alert(alert)
        return fresh

    def _fresh_alerts(self, alerts: tuple[TrainingAlert, ...]) -> tuple[TrainingAlert, ...]:
        active_now = {alert.code for alert in alerts if _is_persistent_alert(alert)}
        fresh: list[TrainingAlert] = []
        for alert in alerts:
            if _is_persistent_alert(alert):
                if alert.code not in self._active_persistent:
                    fresh.append(alert)
                continue
            key = (alert.code, alert.step, alert.evidence)
            if key in self._emitted_events:
                continue
            self._emitted_events.add(key)
            fresh.append(alert)
        self._active_persistent = active_now
        return tuple(fresh)

    def _evaluate(
        self,
        current: TrainingObservation,
        *,
        event_metrics: Mapping[str, float] | None = None,
    ) -> tuple[TrainingAlert, ...]:
        alerts: list[TrainingAlert] = []
        if self.config.detect_non_finite:
            metrics = current.metrics if event_metrics is None else event_metrics
            invalid = tuple(
                f"{name}={value}"
                for name, value in metrics.items()
                if not math.isfinite(value)
            )
            if invalid:
                alerts.append(
                    TrainingAlert(
                        code="non_finite_metric",
                        severity="critical",
                        message="Training produced a non-finite metric.",
                        evidence=invalid,
                        step=current.step,
                    )
                )
        window = self._observations[-self.config.patience :]
        if len(window) >= self.config.patience:
            if self.config.detect_stagnation:
                alerts.extend(self._stagnation_alerts(window, current.step))
            if self.config.detect_overfitting:
                alert = self._overfitting_alert(window, current.step)
                if alert is not None:
                    alerts.append(alert)

        # Built-in detectors only use the bounded patience window above. Avoid
        # copying the entire history on every step when no plugin needs it.
        if self._detectors:
            history = tuple(self._observations)
            for detector in self._detectors:
                alerts.extend(detector(history, current, self.config))
        return tuple(alerts)

    def _stagnation_alerts(
        self, window: list[TrainingObservation], step: int
    ) -> list[TrainingAlert]:
        alerts: list[TrainingAlert] = []
        common_names = set.intersection(*(set(item.metrics) for item in window))
        for name in sorted(common_names):
            if "loss" not in name.lower():
                continue
            values = [item.metrics[name] for item in window]
            if all(math.isfinite(value) for value in values) and (
                max(values) - min(values) <= self.config.min_delta
            ):
                alerts.append(
                    TrainingAlert(
                        code=f"stagnation:{name}",
                        severity="warning",
                        message=f"{name} has not changed meaningfully in the recent window.",
                        evidence=tuple(
                            f"step {item.step}: {name}={item.metrics[name]}" for item in window
                        ),
                        step=step,
                        persistent=True,
                    )
                )
        return alerts

    def _overfitting_alert(
        self, window: list[TrainingObservation], step: int
    ) -> TrainingAlert | None:
        train_name = _first_metric(window, ("train_loss", "training_loss", "loss"))
        validation_name = _first_metric(window, ("validation_loss", "val_loss", "eval_loss"))
        if train_name is None or validation_name is None:
            return None
        train = [item.metrics[train_name] for item in window]
        validation = [item.metrics[validation_name] for item in window]
        delta = self.config.min_delta
        train_falling = all(
            right < left - delta for left, right in zip(train, train[1:], strict=False)
        )
        validation_rising = all(
            right > left + delta
            for left, right in zip(validation, validation[1:], strict=False)
        )
        if not (train_falling and validation_rising):
            return None
        return TrainingAlert(
            code="possible_overfitting",
            severity="warning",
            message="Training and validation loss are diverging.",
            evidence=(
                f"{train_name}: {train[0]} -> {train[-1]}",
                f"{validation_name}: {validation[0]} -> {validation[-1]}",
            ),
            step=step,
            persistent=True,
        )


def _is_persistent_alert(alert: TrainingAlert) -> bool:
    return (
        alert.persistent
        or alert.code == "possible_overfitting"
        or alert.code.startswith("stagnation:")
    )


def _first_metric(
    observations: list[TrainingObservation], candidates: tuple[str, ...]
) -> str | None:
    if not observations:
        return None
    first_names = tuple(observations[0].metrics)
    for candidate in candidates:
        for name in first_names:
            if _normalized_metric_name(name) != candidate:
                continue
            if all(name in item.metrics for item in observations):
                return name
    return None


def _normalized_metric_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
