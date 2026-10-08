# Exportación, limitaciones y solución de problemas

## Exportar

```python
from trainlens import render_report, write_report

markdown = render_report(report, format="markdown")
write_report(report, "report.md")
write_report(report, "report.html")
write_report(report, "report.json")
write_report(report, "report.pdf")
```

El formato se infiere de `.md`, `.markdown`, `.html`, `.json` o `.pdf`, o se
puede indicar. El PDF funciona después de `pip install trainlens`, sin extras.
Los renderers HTML y PDF priorizan
portabilidad sobre soporte completo de CommonMark o maquetación avanzada. JSON
convierte valores flotantes no finitos en `null`.

## Problemas habituales

### No hay una sesión IPython activa

Pasa un espacio de nombres: `build_paper_report(vars(my_module))` o un mapping
preparado. La detección automática solo funciona en IPython/Jupyter.

### El proveedor LLM no está configurado

Define valores no vacíos para `TRAINLENS_LLM_BASE_URL` y
`TRAINLENS_LLM_MODEL`. Define `TRAINLENS_LLM_API_KEY` cuando el endpoint requiera
autenticación; puede omitirse en servidores locales sin autenticación.

### Faltan métricas

Mantén los historiales u objetos trainer en el espacio inspeccionado, o aporta
métricas explícitas con `AnalysisConfig(metrics=...)`. Usa valores numéricos y
finitos. TrainLens no ejecuta el entrenamiento ni consulta servidores remotos de
tracking.

### La dirección de comparación es desconocida

Registra semántica de dominio con `register_metric(MetricSpec(...))` cuando el
nombre quede fuera del vocabulario integrado. Sin dirección registrada, el delta
numérico sigue siendo válido, pero TrainLens no afirmará que sea mejora o
regresión.

### El contexto LLM aparece truncado

`ContextPolicy` limita la evidencia saliente. Usa `preview_notebook_context()`
para revisar el contexto sanitizado exacto y aumenta solo los presupuestos que
necesites. Las omisiones y el truncado global se marcan explícitamente.

### Falla la exportación PDF

ReportLab se incluye en la instalación normal. Si falta (por ejemplo por una
instalación con `--no-deps`), reinstala con
`python -m pip install --upgrade --force-reinstall trainlens`.
Para maquetación editorial
compleja, exporta Markdown o JSON y utiliza una herramienta especializada.

## Limitaciones actuales

- API en fase alpha y compatibilidad declarada con Python 3.11–3.14
- El historial de magics sigue en memoria, aunque los `TrainingRun` terminados
  pueden guardarse y cargarse como JSON
- La detección de frameworks sigue siendo heurística; usa `AnalysisConfig` para
  seleccionar modelo/trainer explícitamente en notebooks ambiguos
- El conjunto integrado de detectores en vivo es pequeño; se pueden añadir
  detectores propios mediante la interfaz pública `AlertDetector`
- Heurísticas deterministas de siguiente paso y Pareto, no inferencia causal ni
  una búsqueda completa de hiperparámetros
- Calidad, privacidad y coste del texto LLM dependientes del proveedor configurado
  o inyectado

Por ello TrainLens es ante todo una ayuda ligera para cuadernos y una capa de
informes, no una plataforma MLOps completa.
