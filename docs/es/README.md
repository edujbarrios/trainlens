# Documentación de TrainLens

TrainLens inspecciona la evidencia de entrenamiento disponible en Python o en
un cuaderno Jupyter y la convierte en informes, comparaciones, alertas y
experimentos de seguimiento revisables. La mayor parte del análisis es local y
determinista; el texto generado por un LLM es opcional.

## Por dónde empezar

- [Instalación y guía rápida](primeros-pasos.md)
- [Flujo en cuadernos y adaptadores](cuadernos.md)
- [API de Python: informes, comparaciones y experimentos](api-python.md)
- [Monitorización y callbacks](monitorizacion.md)
- [Configuración del LLM, prompts y privacidad](llm-y-privacidad.md)
- [Exportación, limitaciones y solución de problemas](exportacion-y-problemas.md)

## Cuándo resulta útil

TrainLens encaja bien cuando el estado de los experimentos vive en variables de
un cuaderno y se necesita un registro ligero y agnóstico al framework sin
desplegar un servidor de seguimiento. Es especialmente útil para comparar unos
pocos entrenamientos, detectar anomalías métricas sencillas y generar informes
consistentes para revisión.

No sustituye a un gestor integral de experimentos, un perfilador, una suite de
evaluación ni un sistema de diagnóstico causal. Sus heurísticas son pequeñas y
explicables a propósito. Hay que contrastar las conclusiones con los datos y el
conocimiento del dominio.

## Requisitos

- Python 3.11 o posterior
- IPython/Jupyter para los comandos mágicos
- Un endpoint compatible con OpenAI solo para informes generados por LLM
- `reportlab` solo para exportar PDF

