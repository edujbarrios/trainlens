"""Dependency-free callback adapters for live TrainLens monitoring."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from trainlens.monitoring import TrainingAlert, TrainingObservation, TrainLensMonitor

ExplanationHandler = Callable[[TrainingObservation], None]


class TrainLensCallback:
    """Feed Keras, Transformers, or Lightning metric updates into a monitor.

    The class intentionally does not inherit from framework callback bases, keeping
    TrainLens free of heavyweight required dependencies.
    """

    def __init__(
        self,
        monitor: TrainLensMonitor | None = None,
        *,
        alerts: bool = True,
        explain_every: int | None = None,
        on_explain: ExplanationHandler | None = None,
        stop_on_anomaly: bool = False,
    ) -> None:
        if explain_every is not None and explain_every < 1:
            raise ValueError("explain_every must be at least 1")
        if explain_every is not None and on_explain is None:
            raise ValueError("on_explain is required when explain_every is configured")
        self.monitor = monitor or TrainLensMonitor()
        self.alerts_enabled = alerts
        self.explain_every = explain_every
        self.on_explain = on_explain
        self.stop_on_anomaly = stop_on_anomaly
        self.alerts: list[TrainingAlert] = []
        self.stop_requested = False
        self.model: Any | None = None

    def set_model(self, model: Any) -> None:
        """Receive the active model from Keras without importing Keras."""

        self.model = model

    def observe(self, step: int, metrics: Mapping[str, Any]) -> tuple[TrainingAlert, ...]:
        """Normalize a framework metric mapping and update the monitor."""

        numeric = _numeric_metrics(metrics)
        detected = self.monitor.observe(step, numeric)
        if self.alerts_enabled:
            self.alerts.extend(detected)
        if self.stop_on_anomaly and any(alert.severity == "critical" for alert in detected):
            self.stop_requested = True
            if self.model is not None and hasattr(self.model, "stop_training"):
                self.model.stop_training = True
        if (
            self.explain_every is not None
            and step % self.explain_every == 0
            and self.on_explain is not None
        ):
            self.on_explain(self.monitor.observations[-1])
        return detected

    def on_epoch_end(self, epoch: int, logs: Mapping[str, Any] | None = None) -> None:
        """Keras-compatible epoch hook."""

        self.observe(epoch, logs or {})

    def on_log(
        self,
        args: Any = None,
        state: Any = None,
        control: Any = None,
        logs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Hugging Face Transformers-compatible logging hook."""

        del args, kwargs
        step = int(getattr(state, "global_step", len(self.monitor.observations)))
        self.observe(step, logs or {})
        if self.stop_requested and control is not None:
            control.should_training_stop = True
        return control

    def on_train_epoch_end(self, trainer: Any, pl_module: Any = None) -> None:
        """PyTorch Lightning-compatible training epoch hook."""

        del pl_module
        step = int(getattr(trainer, "current_epoch", len(self.monitor.observations)))
        metrics = getattr(trainer, "callback_metrics", {})
        self.observe(step, metrics)
        if self.stop_requested and hasattr(trainer, "should_stop"):
            trainer.should_stop = True


def _numeric_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for name, value in metrics.items():
        if isinstance(value, bool):
            continue
        candidate = value.item() if callable(getattr(value, "item", None)) else value
        if isinstance(candidate, int | float):
            normalized[str(name)] = float(candidate)
    return normalized
