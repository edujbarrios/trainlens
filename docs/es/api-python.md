# API de Python

## Informes con LLM

```python
from trainlens import build_improvement_ideas, build_paper_report

paper = build_paper_report(globals(), max_metric_points=12)
ideas = build_improvement_ideas(globals(), max_metric_points=12)
print(paper.markdown)
```

`build_llm_report` es un punto de entrada equivalente al informe científico.
`LiveReport` contiene el Markdown generado y el `AnalysisResult` extraído.
`max_metric_points` debe ser al menos 2; los historiales largos conservan
extremos, mínimo, máximo, cantidad y una muestra ordenada.

## Inspeccionar la configuración de entrenamiento

```python
from trainlens import inspect_training_profile

profile = inspect_training_profile(model, trainer=trainer, namespace=globals())
print(profile.strategy)
print(profile.trainable_components)
print(profile.frozen_components)
print(profile.parameters)
```

`inspect_training_profile()` usa *duck typing* seguro y no importa paquetes
opcionales de frameworks. Normaliza argumentos habituales de Hugging Face,
configuración PEFT/LoRA/DoRA y parámetros de VLM como el tipo de projector,
learning rates multimodales, selección de características visuales y longitud
de secuencia.

En VLMs, TrainLens distingue alineamiento sólo del projector, fine-tuning con
adaptadores y fine-tuning VLM más amplio cuando el estado disponible permite
inferirlo. El perfil también registra componentes de visión, lenguaje y
projector multimodal detectados como entrenables o congelados. La inspección es
*best effort*: un atributo desconocido o inaccesible se omite en vez de hacer
fallar el análisis del notebook.

Los flujos normales de `%explain_training` e informes reutilizan el mismo
perfil para añadir evidencia y recomendaciones específicas de VLM. El perfil
sanitizado exacto que se incluiría en una petición LLM opcional puede verse con
`preview_notebook_context()` o `%explain_training --dry-run`.

## Comparar entrenamientos

```python
from trainlens import compare_runs, render_run_comparison

result = compare_runs(
    {"loss": 0.40, "f1": 0.71}, {"loss": 0.34, "f1": 0.75},
    baseline_name="run-a", experiment_name="run-b",
)
print(render_run_comparison(result))
```

Loss, error, perplexity, WER, CER, latencia, FAD y Fréchet se consideran
mejores al bajar. Accuracy, AUC, F1, precision, recall, score, mAP y NDCG se
consideran mejores al subir. Los nombres desconocidos reciben un delta sin
afirmar que haya mejora. Los valores ausentes o no finitos se tratan de forma
explícita.

## Planificar un experimento controlado

```python
from trainlens import ExperimentRun, experiment_config, suggest_next_experiment

runs = [ExperimentRun(
    name="baseline",
    metrics={"train_loss": 0.20, "validation_loss": 0.50},
    parameters={"learning_rate": 1e-3, "dropout": 0.10, "batch_size": 32},
    estimated_cost="medium",
)]
recommendation = suggest_next_experiment(
    runs, objective_metric="validation_loss", minimum_improvement=0.01
)
next_parameters = experiment_config(
    recommendation, base_parameters=runs[0].parameters
)
```

La recomendación cambia una sola variable e incluye evidencia, confianza,
coste, constantes y un criterio de éxito medible. Es una heurística
determinista, no una garantía. Indica una métrica objetivo con dirección
reconocible si la selección automática no sirve.

## Puntos de entrada públicos

El paquete superior exporta constructores de informes, configuración de
prompts, comparaciones, planificación, monitorización, callbacks,
`TrainingProfile`, `inspect_training_profile()` y exportación. Los componentes
internos de inspección y modelos de los subpaquetes deben considerarse APIs de
nivel inferior.

