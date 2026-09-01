# LLM configuration, prompts, and privacy

TrainLens talks to OpenAI-compatible `/chat/completions` endpoints using the
Python standard library. Configure all required variables before calling an LLM
report helper or magic:

```bash
export TRAINLENS_LLM_BASE_URL="https://api.openai.com/v1"
export TRAINLENS_LLM_API_KEY="replace-me"
export TRAINLENS_LLM_MODEL="your-model"
export TRAINLENS_LLM_TIMEOUT_SECONDS="120"
```

On Windows PowerShell, use `$env:NAME = "value"`. The base URL, API key, and
model are all required. Invalid or non-positive timeout values fall back to 120
seconds. Local OpenAI-compatible servers can use placeholder keys if they
require no authentication but TrainLens still requires the variable to be
non-empty.

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
of detected notebook evidence to the configured provider. TrainLens redacts
likely secrets and truncates large literals before prompt construction, but
redaction is defense in depth—not a reason to place credentials or sensitive
records in notebook variables. Review provider retention and privacy terms.

## Failure modes

Missing configuration raises a provider configuration error instead of silently
falling back to fabricated prose. Network, authentication, model, timeout, and
malformed-response errors are surfaced to the caller.

