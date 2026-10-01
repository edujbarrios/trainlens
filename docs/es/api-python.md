# API de Python

## Selección explícita del análisis

```python
from trainlens import AnalysisConfig, analyze

result = analyze(
    globals(),
    config=AnalysisConfig(
        model="model_v2",
        trainer="trainer_v2",
        strict=True,
        metrics={"validation_loss": validation_loss},
    ),
)
```

`AnalysisConfig` puede seleccionar modelo y trainer por nombre de variable u
objeto concreto. `metrics` y `labels` permiten aportar evidencia autoritativa.
Con `strict=True`, una selección automática ambigua produce un error en vez de
elegir silenciosamente el primer candidato.

## Informes con LLM

```python
from trainlens import ContextPolicy, build_improvement_ideas, build_paper_report

policy = ContextPolicy(
    max_metric_points=12,
    max_metric_series=20,
    max_variables=20,
    max_chars=32_000,
)
paper = build_paper_report(globals(), context_policy=policy)
ideas = build_improvement_ideas(globals(), context_policy=policy)
print(paper.markdown)
```

`build_llm_report` es un punto de entrada equivalente al informe científico.
`LiveReport` contiene el Markdown generado y el `AnalysisResult` extraído. El
argumento histórico `max_metric_points` sigue soportado. `ContextPolicy` añade
presupuestos globales para series de métricas, variables, artefactos de
entrenamiento, candidatos de modelo y tamaño total del contexto. Las omisiones
o truncados quedan indicados explícitamente.

Los informes LLM incluyen el resumen determinista de TrainLens, señales,
provenance de evidencia y recomendaciones. El proveedor explica esos hallazgos
locales en vez de sustituirlos por una interpretación separada del notebook.

### Inyectar un proveedor LLM

```python
from trainlens import LLMConfig, OpenAICompatibleProvider, build_paper_report

provider = OpenAICompatibleProvider(
    LLMConfig(
        base_url="http://localhost:11434/v1",
        model="llama3.2",
    )
)
report = build_paper_report(globals(), provider=provider)
```

`LLMProvider` es un protocolo público, por lo que se pueden inyectar transports
propios. `LLMConfig.api_key` es opcional; el proveedor OpenAI-compatible omite
`Authorization` cuando no hay clave configurada.

## Inspeccionar la configuración de entrenamiento

```python
from trainlens import inspect_training_profile

profile = inspect_training_profile(model, trainer=trainer, namespace=globals())
print(profile.strategy)
print(profile.trainable_parameters, profile.total_parameters)
print(profile.trainable_fraction)
print(profile.adapters, profile.active_adapters)
print(profile.quantization)
```

`inspect_training_profile()` usa *duck typing* seguro y no importa paquetes
opcionales. Normaliza argumentos de Hugging Face, PEFT/LoRA/DoRA y configuración
VLM. Cuando el estado lo permite, distingue fine-tuning completo de parcial,
calcula parámetros entrenables/totales y su fracción, detecta múltiples adapters
PEFT y los activos, y reconoce cuantización habitual de 4/8 bits.

Los flujos de `%explain_training` e informes reutilizan el mismo perfil. El
contexto sanitizado exacto puede verse con `preview_notebook_context()` o
`%explain_training --dry-run`.

## Registrar semántica de métricas propias

```python
from trainlens import MetricSpec, register_metric

register_metric(
    MetricSpec(
        name="dice",
        direction="higher",
        lower_bound=0.0,
        upper_bound=1.0,
        aliases=("dice_score",),
    )
)
```

`MetricRegistry` se comparte entre comparación y planificación experimental. Un
`MetricSpec` puede definir dirección, límites, aliases, unidad y umbrales de
cambio material.

## Runs portables y comparaciones más ricas

```python
from trainlens import compare_runs, load_run, save_run, training_run_from_analysis

run = training_run_from_analysis(result, parameters={"learning_rate": 1e-4})
save_run(run, "run-a.json")
restored = load_run("run-a.json")
comparison = compare_runs(run, restored)
```

El JSON de `TrainingRun` conserva historiales de métricas y steps, además de
parámetros y metadatos portables. Al comparar dos `TrainingRun`, TrainLens
informa de métricas finales, parámetros cambiados, mejores valores de trayectoria
y número de observaciones.

## Planificar experimentos controlados y multiobjetivo

```python
from trainlens import (
    ExperimentRun,
    MetricConstraint,
    ObjectiveSpec,
    pareto_front,
    suggest_multiobjective_experiment,
)

runs = [ExperimentRun(
    name="baseline",
    metrics={"validation_loss": 0.50, "latency_ms": 9.0},
    parameters={"learning_rate": 1e-3, "dropout": 0.10},
)]

objectives = (
    ObjectiveSpec(metric="validation_loss", direction="min"),
    ObjectiveSpec(metric="latency_ms", direction="min"),
)
constraints = (
    MetricConstraint(metric="latency_ms", operator="<=", threshold=10.0),
)
front = pareto_front(runs, objectives, constraints=constraints)
recommendation = suggest_multiobjective_experiment(
    runs,
    objectives,
    constraints=constraints,
)
```

La API multiobjetivo puede aplicar restricciones duras y razonar sobre el frente
de Pareto en vez de convertir todas las métricas en una única puntuación.
`suggest_next_experiment()` sigue disponible para el caso de un objetivo.

## Monitorización extensible

```python
from trainlens import TrainingAlert, TrainLensMonitor


def gradient_detector(history, current, config):
    value = current.metrics.get("gradient_norm")
    if value is None or value <= 10:
        return ()
    return (
        TrainingAlert(
            code="high_gradient_norm",
            severity="warning",
            message="Gradient norm elevado.",
            evidence=(f"gradient_norm={value}",),
            step=current.step,
            persistent=True,
        ),
    )


monitor = TrainLensMonitor(detectors=(gradient_detector,))
```

Los `AlertDetector` personalizados reciben el historial inmutable, la observación
actual y `MonitorConfig`. Las alertas persistentes personalizadas usan el mismo
ciclo de rearme tras recuperación que las alertas integradas.

## Puntos de entrada públicos

El paquete superior exporta configuración explícita del análisis, controles de
informes/proveedores, semántica de métricas, runs portables y comparación,
planificación experimental, monitorización/callbacks, `TrainingProfile`,
`inspect_training_profile()` y exportación. Los componentes internos de
inspección deben considerarse APIs de nivel inferior.
