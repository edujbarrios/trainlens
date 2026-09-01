# Monitoring and callbacks

`TrainLensMonitor` consumes one metric mapping per step and emits immutable,
evidence-backed alerts. Detection is local and deterministic.

```python
from trainlens import MonitorConfig, TrainLensMonitor

monitor = TrainLensMonitor(MonitorConfig(patience=3, min_delta=0.01))
for epoch, metrics in enumerate(training_loop()):
    for alert in monitor.observe(epoch, metrics):
        print(alert.code, alert.severity, alert.evidence)
```

It detects:

- non-finite metrics (`critical`)
- loss stagnation across the configured window (`warning`)
- falling training loss with rising validation loss (`warning`)

`patience` must be an integer of at least 2. `min_delta` must be finite and
non-negative. Individual detectors can be disabled in `MonitorConfig`.

## Receive alerts immediately

```python
monitor = TrainLensMonitor(on_alert=lambda alert: logger.warning(alert.message))
```

The `observations` property returns an immutable view of recorded snapshots.
TrainLens suppresses duplicate alerts with the same code, step, and evidence.

## Framework-shaped callback

```python
from trainlens import TrainLensCallback

callback = TrainLensCallback(
    alerts=True,
    stop_on_anomaly=False,
    explain_every=5,
    on_explain=lambda observation: print(observation),
)

callback.on_epoch_end(epoch, logs)                         # Keras-style
callback.on_log(state=state, control=control, logs=logs)   # Transformers-style
callback.on_train_epoch_end(trainer)                       # Lightning-style
```

Stopping is opt-in and applies only to critical anomalies. `on_explain` is your
own function; the callback never contacts an LLM by itself.

