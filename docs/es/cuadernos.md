# Flujo en cuadernos y adaptadores

TrainLens busca en un espacio de nombres historiales métricos, objetos parecidos
a modelos, etiquetas y metadatos ligeros. Normaliza alias como `eval_loss`,
`val_loss`, `validation_loss`, `train/accuracy`, `eval-accuracy`, `test_loss` y
`test_f1`, manteniendo explícitamente el split de train / validación / test.

## Evidencia compatible

- Diccionarios, secuencias, historiales de logs y eventos tipo traza
- Contenedores por split como `train_metrics`, `validation_metrics` y `test_metrics`
- Objetos de historial de Keras
- `trainer.state.log_history` de Hugging Face
- Métricas registradas y callbacks de PyTorch Lightning
- Objetos tipo modelo, optimizador, scheduler, loader y dataset de PyTorch
- Etiquetas con nombres como `y_train`, `labels` o `target`

Los adaptadores usan duck typing y nombres de módulos; TensorFlow, PyTorch,
Transformers y Lightning no son dependencias obligatorias.

## Resultados de train / validación / test

TrainLens conserva el papel de cada split en vez de mezclar todas las métricas:

```python
history = {
    "train_loss": [1.2, 0.7, 0.4],
    "val_loss": [1.1, 0.72, 0.55],
    "train_accuracy": [0.62, 0.80, 0.91],
    "val_accuracy": [0.60, 0.77, 0.84],
}

test_metrics = {
    "loss": 0.61,
    "accuracy": 0.81,
    "f1": 0.80,
    "precision": 0.82,
    "recall": 0.79,
}

from trainlens import analyze

result = analyze(globals())
result.metrics
```

El resultado estructurado incluye métricas finales como `validation_f1`,
`test_precision` o `test_perplexity`, no solo loss y accuracy. Las reglas de
generalización tienen en cuenta si una métrica debe subir o bajar, por lo que
un run que solo registre loss también puede mostrar señales de posible
overfitting. Una caída material entre validación y test se informa por separado,
sin asumir una causa concreta: puede indicar sobreselección en validación,
diferencias entre splits o distribution shift. Si hay validación pero no test,
TrainLens puede recomendar una evaluación final sobre un conjunto realmente
retenido.

## Curvas de entrenamiento

El plotting es opcional para mantener ligero el paquete base:

```python
%pip install -q "trainlens[plots]"
```

Después:

```python
from trainlens import plot_training_curves, training_curves

plot_training_curves(globals(), metrics=("loss", "accuracy"))
curves = training_curves(globals())
```

`plot_training_curves()` representa train y validación y coloca una métrica de
test escalar en la última observación cuando corresponde. Devuelve la figura de
Matplotlib para poder personalizarla o guardarla. `training_curves()` devuelve
los mismos datos de forma estructurada sin obligarte a dibujarlos.

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

Puedes ver solo la evidencia saneada con:

```python
%explain_training --dry-run
```

o desde Python:

```python
from trainlens import preview_notebook_context

preview_notebook_context()
```

Si además quieres inspeccionar las instrucciones confiables y los mensajes
exactos que recibirá el endpoint OpenAI-compatible:

```python
from trainlens import preview_llm_request

preview = preview_llm_request(
    globals(),
    mode="improvement_ideas",
    provider=llm,
    prompt_options=prompt,
)
print(preview.system_prompt)
print(preview.user_prompt)
```

Ninguna de las dos rutas de previsualización hace una petición de red. Los
historiales incluyen pasos alineados o épocas fraccionarias cuando existen, y
las series largas siguen limitadas por la política de contexto.

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
# Keras
history = model.fit(x_train, y_train, validation_data=(x_val, y_val))
test_metrics = dict(zip(model.metrics_names, model.evaluate(x_test, y_test)))

# Hugging Face
trainer.train()
eval_metrics = trainer.evaluate()
test_output = trainer.predict(test_dataset)
test_metrics = test_output.metrics

# Lightning
trainer.fit(module, datamodule=datamodule)
test_metrics = trainer.test(module, datamodule=datamodule)[0]

# Para PyTorch puro, conserva objetos y resultados visibles:
model, optimizer, scheduler, train_loader
history, test_metrics
```

Cuando hay objetos reales de modelo/trainer, TrainLens también puede inspeccionar
configuración habitual de fine-tuning: parámetros entrenables/congelados,
adaptadores PEFT/LoRA, rank/alpha/target modules, cuantización, learning rates y
ajustes multimodales compatibles.

Conviene mantener las variables descriptivas pequeñas y bien nombradas.
TrainLens evita serializar objetos arbitrarios o datasets completos. Si el
mismo modelo aparece por introspección genérica y por un adaptador de framework,
TrainLens combina la evidencia en un único candidato en lugar de duplicarlo en
el prompt.

## Espacios de nombres explícitos

Fuera de IPython, pasa un mapping:

```python
from trainlens import build_paper_report, preview_notebook_context

namespace = {"history": history, "test_metrics": test_metrics}
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

## Visualización de entrenamientos largos

`plot_training_curves()` destaca ahora el mejor valor de **validación** y
distingue el resultado final de test. Los gráficos de 1.000 o más observaciones
pueden usar una selección min/max para conservar picos y valles sin dibujar
miles de marcadores. El historial completo no se modifica.

```python
from trainlens import plot_training_curves

figura = plot_training_curves(
    {"log_history": trainer.state.log_history},
    metrics=("loss",),
    x_axis="auto",
    max_plot_points=300,
)
```

Con `x_axis="auto"` se usan los steps reales únicamente cuando están completos
en todas las series. Si no, el eje X identifica **observaciones** (no epochs
inventadas). `x_axis="step"` permite conservar huecos donde falta el step;
requiere metadatos de steps para las trayectorias con varios valores.

El [notebook de inicio](../../examples/quickstart.ipynb) contiene un ejemplo
sintético reproducible de 1.000 observaciones.
