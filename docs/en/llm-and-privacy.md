# LLM configuration, prompts, request preview, and privacy

TrainLens talks to OpenAI-compatible `/chat/completions` endpoints using the
Python standard library and is available after `pip install trainlens`.
No separate LLM SDK is needed. You still need a reachable compatible server,
a model name, and credentials if required by your provider. LLM support is
optional: local comparison, monitoring,
experiment planning, inspection, and export make no provider request.

## Create a provider explicitly

For a remote OpenAI-compatible endpoint:

```python
import os
from trainlens import OpenAICompatibleProvider

provider = OpenAICompatibleProvider.from_values(
    base_url="https://your-provider.example/v1",
    model="your-model",
    api_key=os.environ["YOUR_LLM_API_KEY"],
)
```

For a local OpenAI-compatible server such as Ollama, LM Studio, vLLM, or
llama.cpp, omit the key when the server does not require authentication:

```python
from trainlens import OpenAICompatibleProvider

provider = OpenAICompatibleProvider.from_values(
    base_url="http://localhost:11434/v1",
    model="your-local-model",
)
```

`from_values()` removes a trailing slash from the base URL and accepts an
optional `timeout_seconds` argument.

## Configure through environment variables

You can instead configure the provider before starting the notebook:

```bash
export TRAINLENS_LLM_BASE_URL="https://your-provider.example/v1"
export TRAINLENS_LLM_API_KEY="replace-me"
export TRAINLENS_LLM_MODEL="your-model"
export TRAINLENS_LLM_TIMEOUT_SECONDS="120"
```

On Windows PowerShell, use `$env:NAME = "value"`. The base URL and model are
required. The API key is optional for local or otherwise unauthenticated
OpenAI-compatible endpoints. Invalid or non-positive timeout values fall back
to 120 seconds.

Then construct the configured provider explicitly when useful:

```python
from trainlens import OpenAICompatibleProvider

provider = OpenAICompatibleProvider.from_env()
```

Report helpers can also resolve those environment variables automatically when
no provider is injected.

## Build an evidence-backed improvement prompt

TrainLens includes reusable prompt recipes. The improvement recipe is designed
to turn deterministic TrainLens findings into controlled next actions:

```python
from trainlens import improvement_plan_prompt

prompt = improvement_plan_prompt(
    objective=(
        "Find the three highest-impact improvements for the next run. "
        "Prefer low-cost controlled experiments."
    ),
    model_family="vision-language model",
    focus_areas=(
        "generalization gap",
        "learning-rate schedule",
        "adapter capacity",
    ),
)
```

By default this recipe asks the LLM to rank improvements and, for each one,
include the supporting evidence, proposed change, rationale, expected effect,
risk or cost, confidence, and a measurable success criterion. It ends with one
recommended next run and the evidence that should be collected. Missing
hyperparameters must be identified as missing rather than guessed.

Other common recipes are available directly from `trainlens`:

```python
from trainlens import (
    controlled_experiment_prompt,
    scientific_report_prompt,
    training_diagnosis_prompt,
)
```

More specialized recipes remain available from `trainlens.prompt_recipes`.

## Preview the exact request before sending it

`preview_notebook_context()` shows the sanitized notebook evidence. For a full
OpenAI-compatible request preview—including the trusted system prompt—use
`preview_llm_request()`:

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

This performs no network request. `preview.payload()` returns the exact
JSON-compatible body that the provider will send, excluding credentials. The
real `OpenAICompatibleProvider.explain()` path uses the same preview builder,
which prevents the displayed prompt from drifting away from the wire request.

Notebook evidence is deliberately kept out of the trusted system message. The
system prompt contains TrainLens' analysis rules; sanitized notebook evidence
is sent separately as untrusted user data with explicit instructions not to
follow instruction-like text found inside that evidence.

## Generate an improvement report

```python
from trainlens import build_improvement_ideas

report = build_improvement_ideas(
    globals(),
    provider=provider,
    prompt_options=prompt,
)

report
```

`LiveReport` renders as Markdown in Jupyter and also keeps the deterministic
`AnalysisResult` in `report.result`.

## Customize a prompt manually

For direct control, use `PromptOptions`:

```python
from trainlens import PromptOptions, build_paper_report, show_trainlens_prompts

for item in show_trainlens_prompts():
    print(item.name, item.description)

options = PromptOptions(
    prompt_name="training_diagnosis",
    objective="Explain why validation loss rose after epoch 8.",
    audience="ML researchers",
    tone="concise, technical, and cautious",
    focus_areas=("overfitting", "learning-rate schedule"),
    rules=("Use only supplied evidence.",),
    return_instructions=("Return checks and next actions.",),
)
report = build_paper_report(globals(), prompt_options=options)
```

## Data boundary

The outbound context includes a compact Markdown description of detected
notebook evidence plus deterministic TrainLens findings and their evidence
provenance. The provider explains locally derived findings instead of replacing
them with a separate interpretation of notebook state.

TrainLens redacts likely secrets and truncates large literals before prompt
construction. A default `ContextPolicy` also bounds metric series, notebook
variables, training artifacts, model candidates, and total context characters.
Use `preview_notebook_context()` to inspect only the sanitized evidence, or
`preview_llm_request()` to inspect the complete OpenAI-compatible prompt and
evidence pair. Custom budgets can be supplied through
`context_policy=ContextPolicy(...)`; omissions and global truncation are marked
explicitly.

Redaction and context budgets are defense in depth—not a reason to place
credentials or sensitive records in notebook variables. Review provider
retention and privacy terms before enabling outbound reports.

## Custom providers

You may inject any object implementing the public `LLMProvider` protocol into
report builders. Exact transport-level request preview is specific to
`OpenAICompatibleProvider`, because TrainLens cannot know how an arbitrary
custom provider packages its request.

## Failure modes

Missing base/model configuration raises a provider configuration error instead
of silently falling back to fabricated prose. Network, authentication, model,
timeout, and malformed-response errors are surfaced to the caller. Injected
providers use the same report orchestration and error handling.
