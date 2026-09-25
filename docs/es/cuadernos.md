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

%explain_training --name baseline --no-llm
%explain_training --name candidate
%compare_runs baseline candidate
%suggest_improvements
```

`%explain_training` analiza y captura siempre el run local antes de pedir una
explicación a un proveedor LLM. Por tanto, si el proveedor falta o falla, la
captura no se pierde. Usa `--no-llm` para capturar y mostrar el análisis local
sin hacer ninguna petición al proveedor.

Los nombres son etiquetas estables dentro del cuaderno. `%compare_runs
BASELINE EXPERIMENT` acepta nombres o índices de captura empezando en 1. Sin
argumentos conserva el comportamiento anterior y compara las dos capturas más
recientes.

## Previsualizar exactamente qué sale del kernel

Antes de hacer una petición al proveedor, puedes ver el mismo contexto
saneado que TrainLens enviaría:

```python
%explain_training --dry-run
```

o desde Python:

```python
from trainlens import preview_notebook_context

preview_notebook_context()
```

La previsualización usa el mismo constructor de contexto y el mismo saneado de
la ruta real, no hace ninguna petición de red y no requiere configurar un
proveedor. Los historiales incluyen pasos alineados o épocas fraccionarias
cuando existen, y las series largas siguen limitadas por el presupuesto de
puntos métricos.

## Renderizado nativo en el cuaderno

`LiveReport` y las previsualizaciones implementan el protocolo Markdown de
IPython, por lo que pueden ser la última expresión de una celda:

```python
from trainlens import build_paper_report

build_paper_report()
```

El objeto sigue ofreciendo `.markdown` y `.result` para uso programático.

## Ejemplos por framework

```python
history = model.fit(x_train, y_train, validation_data=(x_val, y_val))  # Keras
trainer.train()                                                        # HF
trainer.fit(module, datamodule=datamodule)                             # Lightning

# Para PyTorch puro, conserva estos objetos visibles:
model, optimizer, scheduler, train_loader

%explain_training --name first-run
```

Conviene mantener las variables descriptivas pequeñas y bien nombradas.
TrainLens evita serializar objetos arbitrarios o datasets completos. Si el
mismo modelo aparece por introspección genérica y por un adaptador de framework,
TrainLens combina la evidencia en un único candidato en lugar de duplicarlo en
el prompt.

## Espacios de nombres explícitos

Fuera de IPython, pasa un mapping:

```python
from trainlens import build_paper_report, preview_notebook_context

namespace = {"history": history, "model_name": "classifier-v2"}
preview = preview_notebook_context(namespace)
report = build_paper_report(namespace)
```

Estas funciones sin un espacio de nombres fallan fuera de IPython porque no
existe un espacio de nombres de cuaderno que inspeccionar.

## Recarga de la extensión

Cargar la extensión varias veces es idempotente. Al descargarla se eliminan
los tres comandos mágicos; si se vuelve a cargar en la misma shell de IPython,
se reutiliza la instancia de TrainLens y las capturas locales siguen
disponibles.
