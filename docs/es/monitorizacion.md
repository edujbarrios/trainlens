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

## Callback núcleo para bucles propios

`TrainLensCallback` no depende de frameworks y sigue siendo útil cuando la
aplicación invoca directamente los hooks:

```python
from trainlens import TrainLensCallback

callback = TrainLensCallback(
    alerts=True, stop_on_anomaly=False, explain_every=5,
    on_explain=lambda observation: print(observation),
)
callback.on_epoch_end(epoch, logs)
callback.on_log(state=state, control=control, logs=logs)
callback.on_train_epoch_end(trainer)
```

## Adaptadores nativos de framework

Para registrar TrainLens mediante la API normal de callbacks del framework,
usa la factoría nativa. El framework solo se importa al llamar a la factoría,
por lo que Keras, Transformers y Lightning siguen siendo dependencias
opcionales.

### Keras 3

```python
from trainlens import keras_callback

callback = keras_callback(stop_on_anomaly=True)
model.fit(x, y, callbacks=[callback])
```

El objeto devuelto hereda de `keras.callbacks.Callback`; Keras aporta el ciclo
de vida completo y TrainLens recibe los logs de cada época.

### Hugging Face Transformers

```python
from trainlens import transformers_callback

callback = transformers_callback(stop_on_anomaly=True)
trainer.add_callback(callback)
trainer.train()
```

El objeto hereda de `transformers.TrainerCallback`; `on_log` usa
`state.global_step` y una anomalía crítica puede activar
`control.should_training_stop`.

### Lightning

```python
from trainlens import lightning_callback

callback = lightning_callback(stop_on_anomaly=True)
trainer = Trainer(callbacks=[callback])
trainer.fit(model)
```

El adaptador hereda del callback base de Lightning instalado (`lightning` o
`pytorch-lightning`) y envía `callback_metrics` de cada época a TrainLens.

La parada es opcional y solo se aplica a anomalías críticas. `on_explain` es
una función aportada por la aplicación; el callback no contacta un LLM solo.
