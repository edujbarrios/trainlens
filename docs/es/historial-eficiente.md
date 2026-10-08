# Historiales largos sin desperdiciar tokens

TrainLens **conserva el historial original** de las métricas, incluidos los steps
reales si están disponibles. Al generar contexto para un agente o LLM, no
transmite las 1.000 epochs como texto: resume cada serie con un presupuesto
acotado (12 observaciones por defecto).

## Resumen adaptativo

El resumen es determinista y prioriza:

- el primer y el último valor;
- los mínimos y máximos globales (con al menos 4 puntos de presupuesto);
- cambios locales que una interpolación lineal no explicaría bien.

Los extremos indican su **posición de observación, empezando en 1**. Si existen
steps reales, se conservan; los valores ausentes permanecen como `None`.

```python
from trainlens import load_run, summarize_metric_history

run = load_run("candidate.json")
serie = next(
    m for m in run.metrics if m.name == "loss" and m.split == "validation"
)
resumen = summarize_metric_history(serie, max_points=12)
print(resumen.observations, resumen.minimum_position, resumen.maximum_position)
print(resumen.sampled_positions, resumen.sampled_values)
```

El algoritmo es una aproximación para comunicar evidencia, no una codificación
reversible de la curva. Los históricos completos siguen en `run.metrics`.

## Inspección progresiva de intervalos

El agente puede examinar un intervalo concreto sin incluir toda la curva en el
prompt. Los límites `start` y `end` son inclusivos y se refieren a **posiciones
de observaciones**, no a epochs ni steps inferidos.

```python
from trainlens import inspect_history_window

detalle = inspect_history_window(serie, start=480, end=520, max_points=8)
print(detalle.sampled_positions)
print(detalle.sampled_steps)  # Steps reales o None
```

Esta consulta es local, sin llamadas al proveedor LLM. Para modificar los límites
del contexto en informes y Agent Mode, utiliza
`ContextPolicy(max_metric_points=12, max_chars=...)`. No equivale a un número
exacto de tokens: la tokenización depende del modelo y del contenido.

Se mantienen separadas las métricas de entrenamiento, validación y test; las
decisiones de ajuste deben basarse en entrenamiento/validación, no en test.
