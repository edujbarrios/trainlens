# Flujo en cuadernos y adaptadores

TrainLens busca en un espacio de nombres historiales métricos, objetos parecidos
a modelos, etiquetas y metadatos ligeros. Normaliza alias como `eval_loss`,
`val_loss`, `validation_loss`, `train/accuracy` y `eval-accuracy`.

## Evidencia compatible

- Diccionarios, secuencias, historiales de logs y eventos tipo traza
- Objetos de historial de Keras
- `trainer.state.log_history` de Hugging Face
- Métricas registradas y callbacks de PyTorch Lightning
- Objetos tipo modelo, optimizador, scheduler, loader y dataset de PyTorch
- Etiquetas con nombres como `y_train`, `labels` o `target`

Los adaptadores usan duck typing y nombres de módulos; TensorFlow, PyTorch,
Transformers y Lightning no son dependencias obligatorias.

## Comandos mágicos

```python
%load_ext trainlens.magic.extension
%explain_training
%suggest_improvements
%compare_runs
```

Los dos primeros requieren un endpoint LLM configurado. `%compare_runs` compara
las dos capturas más recientes y avisa si aún no hay dos. Actualmente se ignoran
los argumentos añadidos a estos comandos.

## Ejemplos por framework

```python
history = model.fit(x_train, y_train, validation_data=(x_val, y_val))  # Keras
trainer.train()                                                        # HF
trainer.fit(module, datamodule=datamodule)                             # Lightning

# Para PyTorch puro, conserva estos objetos visibles:
model, optimizer, scheduler, train_loader

%explain_training
```

Conviene mantener las variables descriptivas pequeñas y bien nombradas.
TrainLens evita serializar objetos arbitrarios o datasets completos.

## Espacios de nombres explícitos

Fuera de IPython, pasa un mapping:

```python
from trainlens import build_paper_report

report = build_paper_report({"history": history, "model_name": "classifier-v2"})
```

`build_paper_report()` sin argumentos falla fuera de IPython porque no existe
un espacio de nombres de cuaderno que inspeccionar.

