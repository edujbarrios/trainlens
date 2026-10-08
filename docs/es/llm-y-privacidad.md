# Configuración del LLM, prompts, previsualización y privacidad

TrainLens usa endpoints OpenAI-compatible `/chat/completions`, sin necesitar
un SDK adicional tras `pip install trainlens`. Los endpoints, modelos y
credenciales externos no se incluyen; debes configurarlos cuando corresponda.
El uso de LLM es
opcional: comparación, monitorización, planificación de experimentos,
inspección y exportación locales no hacen ninguna petición al proveedor.

## Crear un proveedor explícitamente

Para un endpoint remoto OpenAI-compatible:

```python
import os
from trainlens import OpenAICompatibleProvider

provider = OpenAICompatibleProvider.from_values(
    base_url="https://tu-proveedor.example/v1",
    model="tu-modelo",
    api_key=os.environ["TU_LLM_API_KEY"],
)
```

Para un servidor local compatible, como Ollama, LM Studio, vLLM o llama.cpp,
normalmente puedes omitir la clave:

```python
from trainlens import OpenAICompatibleProvider

provider = OpenAICompatibleProvider.from_values(
    base_url="http://localhost:11434/v1",
    model="tu-modelo-local",
)
```

`from_values()` normaliza la barra final de la URL y acepta también
`timeout_seconds`.

## Configurar mediante variables de entorno

```bash
export TRAINLENS_LLM_BASE_URL="https://tu-proveedor.example/v1"
export TRAINLENS_LLM_API_KEY="reemplazar"
export TRAINLENS_LLM_MODEL="tu-modelo"
export TRAINLENS_LLM_TIMEOUT_SECONDS="120"
```

En PowerShell usa `$env:NOMBRE = "valor"`. La URL y el modelo son obligatorios.
La clave API es opcional para endpoints locales o sin autenticación. Un timeout
inválido o no positivo vuelve a 120 segundos.

Puedes construir el proveedor configurado de forma explícita:

```python
from trainlens import OpenAICompatibleProvider

provider = OpenAICompatibleProvider.from_env()
```

Los helpers de informes también pueden resolver estas variables automáticamente
cuando no se inyecta un proveedor.

## Crear un prompt de mejoras basado en evidencia

```python
from trainlens import improvement_plan_prompt

prompt = improvement_plan_prompt(
    objective=(
        "Encuentra las tres mejoras de mayor impacto para la siguiente ejecución. "
        "Prioriza cambios controlados y de bajo coste."
    ),
    model_family="vision-language model",
    focus_areas=(
        "generalization gap",
        "learning-rate schedule",
        "adapter capacity",
    ),
)
```

La receta de mejoras pide por defecto, para cada propuesta: evidencia, cambio
concreto, razonamiento, efecto esperado, coste o riesgo, confianza y un criterio
de éxito medible. Termina con una ejecución siguiente recomendada y qué
evidencia conviene recoger. Si falta un hiperparámetro, el modelo debe marcarlo
como ausente en vez de inventarlo.

Las recetas más habituales están disponibles directamente desde `trainlens`:

```python
from trainlens import (
    controlled_experiment_prompt,
    scientific_report_prompt,
    training_diagnosis_prompt,
)
```

Las recetas más especializadas siguen disponibles en
`trainlens.prompt_recipes`.

## Ver exactamente qué se enviará

`preview_notebook_context()` muestra solo la evidencia sanitizada. Para ver la
petición OpenAI-compatible completa, incluido el system prompt de confianza,
usa `preview_llm_request()`:

```python
from trainlens import preview_llm_request

preview = preview_llm_request(
    globals(),
    mode="improvement_ideas",
    provider=provider,
    prompt_options=prompt,
)

print(preview.endpoint)
print(preview.model)
print(preview.system_prompt)
print(preview.user_prompt)
```

Esto no hace ninguna petición de red. `preview.payload()` devuelve el cuerpo
JSON-compatible exacto que usará el proveedor, sin credenciales. La ruta real de
`OpenAICompatibleProvider.explain()` utiliza el mismo constructor de preview,
por lo que el prompt mostrado y el enviado no pueden divergir por caminos de
código distintos.

La evidencia del notebook se mantiene fuera del mensaje de sistema de confianza.
El system prompt contiene las reglas de análisis de TrainLens y la evidencia
sanitizada viaja por separado como datos no confiables, con instrucciones
explícitas para no obedecer texto con apariencia de instrucciones encontrado en
esa evidencia.

## Generar el informe de mejoras

```python
from trainlens import build_improvement_ideas

report = build_improvement_ideas(
    globals(),
    provider=provider,
    prompt_options=prompt,
)

report
```

`LiveReport` se renderiza como Markdown en Jupyter y conserva el
`AnalysisResult` determinista en `report.result`.

## Personalizar un prompt manualmente

```python
from trainlens import PromptOptions, build_paper_report, show_trainlens_prompts

for item in show_trainlens_prompts():
    print(item.name, item.description)

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

## Frontera de datos

El contexto enviado incluye una descripción Markdown compacta de la evidencia
detectada, junto con los hallazgos deterministas de TrainLens y su provenance.
El proveedor explica hallazgos derivados localmente en lugar de sustituirlos por
una interpretación independiente del notebook.

TrainLens redacta posibles secretos y trunca literales grandes antes de construir
el prompt. `ContextPolicy` limita por defecto series de métricas, variables del
notebook, artefactos de entrenamiento, candidatos de modelo y tamaño total del
contexto. Usa `preview_notebook_context()` para revisar solo la evidencia o
`preview_llm_request()` para inspeccionar el prompt y la evidencia completos.
`context_policy=ContextPolicy(...)` permite ajustar los presupuestos y las
omisiones o truncados se marcan explícitamente.

La redacción y los presupuestos son defensa en profundidad: no deben ponerse
credenciales ni datos sensibles en variables del notebook. Revisa las políticas
de retención y privacidad del proveedor antes de activar informes externos.

## Proveedores personalizados

Puedes inyectar cualquier objeto que implemente el protocolo público
`LLMProvider`. La previsualización exacta a nivel de transporte es específica de
`OpenAICompatibleProvider`, porque TrainLens no puede saber cómo empaqueta su
petición un proveedor personalizado arbitrario.

## Errores

La ausencia de URL/modelo y los errores de red, autenticación, modelo, timeout o
respuesta mal formada se muestran al llamador; no se genera texto ficticio. Los
proveedores inyectados usan la misma orquestación y manejo de errores.
