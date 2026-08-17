"""Framework-neutral, real-time monitoring for training metrics."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
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


class TrainLensMonitor:
    """Observe metric updates and emit deterministic alerts as training runs."""

    def __init__(
        self,
        config: MonitorConfig | None = None,
        *,
        on_alert: AlertHandler | None = None,
    ) -> None:
        self.config = config or MonitorConfig()
        self._on_alert = on_alert
        self._observations: list[TrainingObservation] = []
        self._emitted: set[tuple[str, int]] = set()

    @property
    def observations(self) -> tuple[TrainingObservation, ...]:
        """Return immutable access to observations received so far."""

        return tuple(self._observations)

    def observe(self, step: int, metrics: Mapping[str, float | int]) -> tuple[TrainingAlert, ...]:
        """Record one metric snapshot and return alerts triggered by it."""

        if step < 0:
            raise ValueError("step cannot be negative")
        normalized = {name: float(value) for name, value in metrics.items()}
        observation = TrainingObservation(step=step, metrics=MappingProxyType(normalized))
        self._observations.append(observation)

        alerts = self._evaluate(observation)
        fresh = tuple(alert for alert in alerts if self._mark_fresh(alert))
        if self._on_alert is not None:
            for alert in fresh:
                self._on_alert(alert)
        return fresh

    def _mark_fresh(self, alert: TrainingAlert) -> bool:
        key = (alert.code, alert.step)
        if key in self._emitted:
            return False
        self._emitted.add(key)
        return True

    def _evaluate(self, current: TrainingObservation) -> tuple[TrainingAlert, ...]:
        alerts: list[TrainingAlert] = []
        if self.config.detect_non_finite:
            invalid = tuple(
                f"{name}={value}"
                for name, value in current.metrics.items()
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
        if len(window) < self.config.patience:
            return tuple(alerts)
        if self.config.detect_stagnation:
            alerts.extend(self._stagnation_alerts(window, current.step))
        if self.config.detect_overfitting:
            alert = self._overfitting_alert(window, current.step)
            if alert is not None:
                alerts.append(alert)
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
        )


def _first_metric(
    observations: list[TrainingObservation], candidates: tuple[str, ...]
) -> str | None:
    for candidate in candidates:
        if all(candidate in item.metrics for item in observations):
            return candidate
    return None
