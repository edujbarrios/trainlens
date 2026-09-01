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
puede indicar. PDF requiere `trainlens[pdf]`. Los renderers HTML y PDF priorizan
portabilidad sobre soporte completo de CommonMark o maquetación avanzada. JSON
convierte valores flotantes no finitos en `null`.

## Problemas habituales

### No hay una sesión IPython activa

Pasa un espacio de nombres: `build_paper_report(vars(my_module))` o un mapping
preparado. La detección automática solo funciona en IPython/Jupyter.

### El proveedor LLM no está configurado

Define valores no vacíos para `TRAINLENS_LLM_BASE_URL`,
`TRAINLENS_LLM_API_KEY` y `TRAINLENS_LLM_MODEL`.

### Faltan métricas

Mantén los historiales u objetos trainer en el espacio inspeccionado. Usa
nombres convencionales y valores numéricos finitos. TrainLens no ejecuta el
entrenamiento ni consulta servidores remotos de tracking.

### La dirección de comparación es desconocida

El nombre queda fuera del vocabulario integrado. El delta numérico sigue siendo
válido; decide la dirección deseada en código específico del dominio.

### Falla la exportación PDF

Instala `python -m pip install "trainlens[pdf]"`. Para maquetación editorial
compleja, exporta Markdown o JSON y utiliza una herramienta especializada.

## Limitaciones actuales

- API en fase alpha y requisito de Python 3.11+
- Historial de magics en memoria, sin backend persistente ni interfaz colaborativa
- Detección de frameworks y dirección métrica mediante heurísticas
- Tres detectores en vivo, no un análisis estadístico de drift
- Reglas deterministas, no búsqueda causal ni de hiperparámetros
- Calidad y coste del texto LLM dependientes del proveedor

Por ello TrainLens es ante todo una ayuda ligera para cuadernos y una capa de
informes, no una plataforma MLOps completa.

