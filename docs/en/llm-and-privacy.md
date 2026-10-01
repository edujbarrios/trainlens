# LLM configuration, prompts, and privacy

TrainLens talks to OpenAI-compatible `/chat/completions` endpoints using the
Python standard library. Configure a base URL and model before calling an LLM
report helper or magic:

```bash
export TRAINLENS_LLM_BASE_URL="https://api.openai.com/v1"
export TRAINLENS_LLM_API_KEY="replace-me"
export TRAINLENS_LLM_MODEL="your-model"
export TRAINLENS_LLM_TIMEOUT_SECONDS="120"
```

On Windows PowerShell, use `$env:NAME = "value"`. The base URL and model are
required. The API key is optional for local or otherwise unauthenticated
OpenAI-compatible endpoints; TrainLens omits the `Authorization` header when no
key is configured. Invalid or non-positive timeout values fall back to 120
seconds.

You may also inject any object implementing the public `LLMProvider` protocol.
`OpenAICompatibleProvider(LLMConfig(...))` is available when explicit
configuration is preferable to environment variables.

## Select and customize a prompt

```python
from trainlens import PromptOptions, build_paper_report, show_trainlens_prompts

for prompt in show_trainlens_prompts():
    print(prompt.name, prompt.description)

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

Reusable factories live in `trainlens.prompt_recipes` for scientific reports,
diagnostics, overfitting review, improvements, and controlled experiments.

## Data boundary

Local comparison, monitoring, experiment planning, inspection, and export do
not call an external service. LLM helpers send a compact Markdown description
of detected notebook evidence to the configured provider. The outbound context
includes deterministic TrainLens findings and their evidence provenance, so the
provider explains locally derived conclusions instead of independently replacing
them.

TrainLens redacts likely secrets and truncates large literals before prompt
construction. A default `ContextPolicy` also bounds metric series, notebook
variables, training artifacts, model candidates, and total context characters.
Use `preview_notebook_context()` to inspect the exact sanitized evidence. Custom
budgets can be supplied through `context_policy=ContextPolicy(...)`; omissions
and global truncation are marked explicitly.

Redaction and context budgets are defense in depth—not a reason to place
credentials or sensitive records in notebook variables. Review provider
retention and privacy terms before enabling outbound reports.

## Failure modes

Missing base/model configuration raises a provider configuration error instead
of silently falling back to fabricated prose. Network, authentication, model,
timeout, and malformed-response errors are surfaced to the caller. Injected
providers use the same report orchestration and error handling.
