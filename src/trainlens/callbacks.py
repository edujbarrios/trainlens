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
        if explain_every is not None and (
            isinstance(explain_every, bool) or not isinstance(explain_every, int)
        ):
            raise TypeError("explain_every must be an integer or None")
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
        self.params: dict[str, Any] = {}

    def set_model(self, model: Any) -> None:
        """Receive the active model from Keras without importing Keras."""

        self.model = model

    def set_params(self, params: Mapping[str, Any]) -> None:
        """Receive callback parameters from Keras without importing Keras."""

        self.params = dict(params)

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
            and len(self.monitor.observations) % self.explain_every == 0
            and self.on_explain is not None
        ):
            self.on_explain(self.monitor.observations[-1])
        return detected

    def on_epoch_end(self, epoch: Any, *args: Any, **kwargs: Any) -> Any:
        """Handle Keras epoch metrics or a Transformers lifecycle event."""

        if isinstance(epoch, int) and not isinstance(epoch, bool):
            logs = kwargs.get("logs")
            if args and isinstance(args[0], Mapping):
                logs = args[0]
            self.observe(epoch, logs if isinstance(logs, Mapping) else {})
            return None
        return _event_control((epoch, *args), kwargs)

    def on_init_end(self, *args: Any, **kwargs: Any) -> Any:
        return _event_control(args, kwargs)

    def on_train_begin(self, *args: Any, **kwargs: Any) -> Any:
        return _event_control(args, kwargs)

    def on_train_end(self, *args: Any, **kwargs: Any) -> Any:
        return _event_control(args, kwargs)

    def on_epoch_begin(self, *args: Any, **kwargs: Any) -> Any:
        return _event_control(args, kwargs)

    def on_evaluate(self, *args: Any, **kwargs: Any) -> Any:
        return _event_control(args, kwargs)

    def on_save(self, *args: Any, **kwargs: Any) -> Any:
        return _event_control(args, kwargs)

    def on_prediction_step(self, *args: Any, **kwargs: Any) -> Any:
        return _event_control(args, kwargs)

    def on_step_begin(self, *args: Any, **kwargs: Any) -> Any:
        return _event_control(args, kwargs)

    def on_step_end(self, *args: Any, **kwargs: Any) -> Any:
        return _event_control(args, kwargs)

    def on_substep_end(self, *args: Any, **kwargs: Any) -> Any:
        return _event_control(args, kwargs)

    def on_train_batch_begin(self, *args: Any, **kwargs: Any) -> None:
        return None

    def on_train_batch_end(self, *args: Any, **kwargs: Any) -> None:
        return None

    def on_test_begin(self, *args: Any, **kwargs: Any) -> None:
        return None

    def on_test_end(self, *args: Any, **kwargs: Any) -> None:
        return None

    def on_test_batch_begin(self, *args: Any, **kwargs: Any) -> None:
        return None

    def on_test_batch_end(self, *args: Any, **kwargs: Any) -> None:
        return None

    def on_predict_begin(self, *args: Any, **kwargs: Any) -> None:
        return None

    def on_predict_end(self, *args: Any, **kwargs: Any) -> None:
        return None

    def on_predict_batch_begin(self, *args: Any, **kwargs: Any) -> None:
        return None

    def on_predict_batch_end(self, *args: Any, **kwargs: Any) -> None:
        return None

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
        item = getattr(value, "item", None)
        if callable(item):
            try:
                candidate = item()
            except (TypeError, ValueError, RuntimeError):
                continue
        else:
            candidate = value
        if isinstance(candidate, int | float) and not isinstance(candidate, bool):
            normalized[str(name)] = float(candidate)
    return normalized


def _event_control(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Any:
    if "control" in kwargs:
        return kwargs["control"]
    return args[2] if len(args) >= 3 else None
