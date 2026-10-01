# Configuración del LLM, prompts y privacidad

TrainLens usa endpoints OpenAI-compatible `/chat/completions`. Configura una URL
base y un modelo antes de ejecutar un informe o comando mágico con LLM:

```bash
export TRAINLENS_LLM_BASE_URL="https://api.openai.com/v1"
export TRAINLENS_LLM_API_KEY="reemplazar"
export TRAINLENS_LLM_MODEL="tu-modelo"
export TRAINLENS_LLM_TIMEOUT_SECONDS="120"
```

En PowerShell usa `$env:NOMBRE = "valor"`. La URL y el modelo son obligatorios.
La clave API es opcional para endpoints locales o sin autenticación; TrainLens
omite la cabecera `Authorization` cuando no hay clave configurada. Un timeout
inválido o no positivo vuelve a 120 segundos.

También puedes inyectar cualquier objeto que implemente el protocolo público
`LLMProvider`. `OpenAICompatibleProvider(LLMConfig(...))` permite configurar el
proveedor explícitamente sin depender de variables de entorno.

## Elegir y personalizar un prompt

```python
from trainlens import PromptOptions, build_paper_report, show_trainlens_prompts

for prompt in show_trainlens_prompts():
    print(prompt.name, prompt.description)

options = PromptOptions(
    prompt_name="training_diagnosis",
    objective="Explica por qué subió validation loss tras la época 8.",
    audience="investigadores de ML",
    tone="conciso, técnico y prudente",
    focus_areas=("overfitting", "learning-rate schedule"),
    rules=("Usa solamente la evidencia proporcionada.",),
    return_instructions=("Devuelve comprobaciones y siguientes pasos.",),
)
report = build_paper_report(globals(), prompt_options=options)
```

`trainlens.prompt_recipes` incluye factorías para informes científicos,
diagnóstico, overfitting, mejoras y experimentos controlados.

## Frontera de datos

Comparación, monitorización, planificación, inspección y exportación locales no
llaman servicios externos. Los helpers LLM envían al proveedor una descripción
Markdown compacta de la evidencia detectada. El contexto incluye los hallazgos
deterministas de TrainLens y su provenance, de modo que el proveedor explica
conclusiones derivadas localmente en vez de sustituirlas por un análisis
independiente.

TrainLens redacta posibles secretos y trunca literales grandes antes de construir
el prompt. Además, `ContextPolicy` limita por defecto series de métricas,
variables del notebook, artefactos de entrenamiento, candidatos de modelo y el
tamaño total del contexto. `preview_notebook_context()` permite revisar la
evidencia sanitizada exacta; `context_policy=ContextPolicy(...)` permite ajustar
los presupuestos. Las omisiones y truncados globales se indican explícitamente.

La redacción y los presupuestos son defensa en profundidad: no deben ponerse
credenciales ni datos sensibles en variables del notebook. Revisa las políticas
de retención y privacidad del proveedor antes de activar informes externos.

La ausencia de URL/modelo y los errores de red, autenticación, modelo, timeout o
respuesta mal formada se muestran al llamador; no se genera texto ficticio. Los
proveedores inyectados usan la misma orquestación y manejo de errores.
