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

## Core callback for custom loops

`TrainLensCallback` is dependency-free and remains useful when you invoke the
framework-shaped hooks yourself:

```python
from trainlens import TrainLensCallback

callback = TrainLensCallback(
    alerts=True,
    stop_on_anomaly=False,
    explain_every=5,
    on_explain=lambda observation: print(observation),
)

callback.on_epoch_end(epoch, logs)
callback.on_log(state=state, control=control, logs=logs)
callback.on_train_epoch_end(trainer)
```

## Framework-native adapters

For registration through a framework's normal callback API, use the native
factory. Each factory imports its framework only when called, so Keras,
Transformers, and Lightning remain optional dependencies.

### Keras 3

```python
from trainlens import keras_callback

callback = keras_callback(stop_on_anomaly=True)
model.fit(x, y, callbacks=[callback])
```

The returned object subclasses `keras.callbacks.Callback`, so Keras supplies
its full callback lifecycle while epoch logs are delegated to TrainLens.

### Hugging Face Transformers

```python
from trainlens import transformers_callback

callback = transformers_callback(stop_on_anomaly=True)
trainer.add_callback(callback)
trainer.train()
```

The returned object subclasses `transformers.TrainerCallback`; `on_log` uses
`state.global_step`, and a critical anomaly can set
`control.should_training_stop`.

### Lightning

```python
from trainlens import lightning_callback

callback = lightning_callback(stop_on_anomaly=True)
trainer = Trainer(callbacks=[callback])
trainer.fit(model)
```

The adapter subclasses the installed Lightning callback base (`lightning` or
`pytorch-lightning`) and forwards epoch `callback_metrics` to TrainLens.

Stopping is opt-in and applies only to critical anomalies. `on_explain` is your
own function; the callback never contacts an LLM by itself.
