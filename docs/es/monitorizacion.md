# Monitorización y callbacks

`TrainLensMonitor` consume un mapping métrico por paso y emite alertas
inmutables con evidencia. La detección es local y determinista.

```python
from trainlens import MonitorConfig, TrainLensMonitor

monitor = TrainLensMonitor(MonitorConfig(patience=3, min_delta=0.01))
for epoch, metrics in enumerate(training_loop()):
    for alert in monitor.observe(epoch, metrics):
        print(alert.code, alert.severity, alert.evidence)
```

Detecta valores no finitos (`critical`), estancamiento de losses (`warning`) y
loss de entrenamiento descendente junto a loss de validación ascendente
(`warning`). `patience` debe ser un entero de al menos 2 y `min_delta` debe ser
finito y no negativo. Cada detector puede desactivarse en `MonitorConfig`.

## Recibir alertas inmediatamente

```python
monitor = TrainLensMonitor(on_alert=lambda alert: logger.warning(alert.message))
```

`observations` ofrece una vista inmutable de las capturas. Se suprimen alertas
duplicadas con el mismo código, paso y evidencia.

## Callback compatible con varios frameworks

```python
from trainlens import TrainLensCallback

callback = TrainLensCallback(
    alerts=True, stop_on_anomaly=False, explain_every=5,
    on_explain=lambda observation: print(observation),
)
callback.on_epoch_end(epoch, logs)                         # Keras
callback.on_log(state=state, control=control, logs=logs)   # Transformers
callback.on_train_epoch_end(trainer)                       # Lightning
```

La parada es opcional y solo se aplica a anomalías críticas. `on_explain` es
una función aportada por la aplicación; el callback no contacta un LLM solo.

