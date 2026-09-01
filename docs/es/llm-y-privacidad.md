# Configuración del LLM, prompts y privacidad

TrainLens usa endpoints OpenAI-compatible `/chat/completions`. Configura estas
variables antes de ejecutar un informe o comando mágico con LLM:

```bash
export TRAINLENS_LLM_BASE_URL="https://api.openai.com/v1"
export TRAINLENS_LLM_API_KEY="reemplazar"
export TRAINLENS_LLM_MODEL="tu-modelo"
export TRAINLENS_LLM_TIMEOUT_SECONDS="120"
```

En PowerShell usa `$env:NOMBRE = "valor"`. URL, clave y modelo son obligatorios.
Un timeout inválido o no positivo vuelve a 120 segundos. Un servidor local sin
autenticación puede usar una clave ficticia, pero no vacía.

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
Markdown compacta de la evidencia detectada. TrainLens redacta posibles secretos
y trunca literales grandes, pero esto es una defensa adicional: no deben ponerse
credenciales ni datos sensibles en variables del cuaderno. Revisa las políticas
de retención y privacidad del proveedor.

La configuración ausente y los errores de red, autenticación, modelo, timeout o
respuesta mal formada se muestran al llamador; no se genera texto ficticio.

