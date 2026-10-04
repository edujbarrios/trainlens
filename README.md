# TrainLens

**Understand a fine-tuning run, see where it is failing, and decide what to try next — from the notebook.**

TrainLens reads training, validation, and held-out test results; understands common
fine-tuning evidence from Hugging Face, PEFT/LoRA, Keras, Lightning, and PyTorch-style
workflows; plots learning curves; compares runs; and can optionally ask an
OpenAI-compatible LLM to turn the local evidence into an improvement plan.

<p align="center">
  <a href="https://github.com/edujbarrios/trainlens/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/edujbarrios/trainlens/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://pypi.org/project/trainlens/0.14.2/"><img alt="PyPI 0.14.2" src="https://img.shields.io/badge/pypi-0.14.2-blue?logo=pypi"></a>
  <a href="pyproject.toml"><img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-blue"></a>
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-yellow"></a>
</p>

## Try the full notebook workflow

<a href="https://colab.research.google.com/github/edujbarrios/trainlens/blob/main/examples/quickstart.ipynb">
  <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
</a>

Install TrainLens with plotting support:

```python
%pip install -q "trainlens[plots]"
```

Imagine these are the results from a small LoRA/PEFT fine-tuning run. TrainLens accepts
normal notebook variables, so you do not need an experiment tracker just to inspect them:

```python
history = {
    "train_loss":     [1.31, 0.93, 0.66, 0.47, 0.34, 0.25],
    "val_loss":       [1.28, 0.91, 0.70, 0.58, 0.55, 0.57],
    "train_accuracy": [0.55, 0.68, 0.79, 0.87, 0.92, 0.95],
    "val_accuracy":   [0.54, 0.66, 0.75, 0.80, 0.82, 0.81],
}

test_metrics = {
    "loss": 0.64,
    "accuracy": 0.75,
    "f1": 0.74,
    "precision": 0.77,
    "recall": 0.72,
}
```

Analyze the run and plot the curves:

```python
from trainlens import analyze, plot_training_curves, render_report

analysis = analyze(globals())
plot_training_curves(globals())
print(render_report(analysis))
```

TrainLens keeps the split semantics instead of flattening everything into one metric bag:
`train_*`, `validation_*`, and `test_*` remain distinct. It can flag a train/validation
generalization gap, notice when the held-out test split is materially worse than validation,
and keep F1/precision/recall/perplexity-style final metrics in the structured report.

When a real fine-tuning model/trainer is present, TrainLens can also inspect evidence such as
trainable vs. frozen parameters, PEFT adapters, LoRA rank/alpha/targets, quantization,
base/projector/vision learning rates, and common VLM fine-tuning strategies.

## Ask an LLM what happened and what to try next

The LLM is optional. TrainLens does the deterministic analysis locally first. The model gets
a bounded, redacted summary of that evidence and is asked to explain and prioritize it.

### 1. Load your model without putting the API key in the notebook

```python
from getpass import getpass
from trainlens import OpenAICompatibleProvider

llm = OpenAICompatibleProvider.from_values(
    base_url="https://your-provider.example/v1",
    model="your-model",
    api_key=getpass("LLM API key: "),
)
```

For a local OpenAI-compatible server such as Ollama, LM Studio, vLLM, or llama.cpp, the same
API works and an API key is usually unnecessary:

```python
llm = OpenAICompatibleProvider.from_values(
    base_url="http://localhost:11434/v1",
    model="your-local-model",
)
```

### 2. Tell the LLM exactly what kind of training analysis you want

```python
from trainlens import improvement_plan_prompt, prompt_text

prompt = improvement_plan_prompt(
    objective=prompt_text("""
        Explain what happened during this fine-tuning run and propose the three
        highest-value next experiments.

        Use the test split only as final generalization evidence; do not tune to it.
        Prefer controlled, low-cost changes first.
    """),
    model_family="LLM fine-tuning with PEFT/LoRA",
    focus_areas=(
        "train/validation generalization gap",
        "validation-to-test degradation",
        "checkpoint selection and early stopping",
        "learning-rate schedule",
        "regularization and adapter capacity",
    ),
)
```

`prompt_text()` is a small convenience for long notebook instructions: write normal
triple-quoted multiline text, keep it indented with the surrounding Python, and TrainLens
removes the common indentation and outer blank space before the prompt is used. This avoids
repeating quotes and implicit string concatenation on every line.

For prompts that are longer, reused across notebooks, or easier to review separately, keep
them in UTF-8 `.md` files and load them directly:

```markdown
# prompts/improvement_plan.md

Explain what happened during this fine-tuning run and propose the three highest-value next experiments.

Use the test split only as final generalization evidence; do not tune to it.

Prefer:
- controlled experiments
- low-cost changes first
- measurable hypotheses and success criteria
```

```python
from trainlens import load_prompt

prompt = improvement_plan_prompt(
    objective=load_prompt("prompts/improvement_plan.md"),
    model_family="LLM fine-tuning with PEFT/LoRA",
)
```

`load_prompt()` preserves Markdown structure and relative indentation, strips only outer
whitespace, and rejects non-`.md` paths so prompt files stay explicit and easy to version.
The repository also includes `examples/prompts/improvement_plan.md` as a ready-to-copy example.

The improvement recipe asks for the evidence behind every idea, the proposed change,
expected effect, cost/risk, confidence, a measurable success criterion, and one concrete
next run. Missing hyperparameters must be reported as missing rather than invented.

### 3. Inspect exactly what will be sent

```python
from trainlens import preview_llm_request

preview = preview_llm_request(
    globals(),
    mode="improvement_ideas",
    provider=llm,
    prompt_options=prompt,
)

print(preview.system_prompt)  # trusted TrainLens instructions
print(preview.user_prompt)    # sanitized notebook evidence
```

`preview_llm_request()` makes no network request. It uses the same request-building path as
the real provider call, excluding the API key.

### 4. Generate the explanation and next-step report

```python
from trainlens import build_improvement_ideas

report = build_improvement_ideas(
    globals(),
    provider=llm,
    prompt_options=prompt,
)

report  # native Markdown display in Jupyter
```

A useful report should explain the observed train/validation/test behavior, distinguish
observations from hypotheses, rank improvement ideas, and finish with controlled experiments
and measurable success criteria.

## Compare two runs too

For a quick baseline-vs-experiment comparison, no LLM or plotting dependency is required:

```python
from trainlens import compare_runs

comparison = compare_runs(
    {"loss": 0.52, "accuracy": 0.84},
    {"loss": 0.41, "accuracy": 0.89},
    baseline_name="baseline",
    experiment_name="new run",
)

comparison  # renders as Markdown in Jupyter
```

Keep the structured result or export it elsewhere:

```python
from trainlens import render_report

comparison.improvements
comparison.regressions
html = render_report(comparison, format="html")
json_text = render_report(comparison, format="json")
```

TrainLens also understands richer `TrainingRun` objects with metric trajectories and
configuration changes.

## What else can TrainLens do?

- **Analyze train / validation / test results** without collapsing their roles.
- **Inspect fine-tuning configuration** including PEFT/LoRA, trainable parameters,
  quantization, frozen components, and common multimodal settings.
- **Plot learning curves** with `plot_training_curves()` via the optional `plots` extra.
- **Compare runs** with metric-aware improvement/regression semantics.
- **Monitor training** with built-in and extensible alert detectors.
- **Plan controlled next experiments** with objectives, constraints, and Pareto-aware helpers.
- **Author prompts inline or in Markdown files** with `prompt_text()` and `load_prompt()`.
- **Save and load portable runs** for repeatable comparisons.
- **Export reports** to Markdown, JSON, HTML, and optional PDF.
- **Use an optional LLM** to explain deterministic findings and propose evidence-backed next
  steps through any OpenAI-compatible endpoint.

Local analysis is deterministic and does not make an LLM request.

## Deeper API

- [Getting started](docs/en/getting-started.md)
- [Notebook workflows](docs/en/notebooks.md)
- [Python API](docs/en/python-api.md)
- [Monitoring](docs/en/monitoring.md)
- [LLM setup and privacy](docs/en/llm-and-privacy.md)
- [Exports and troubleshooting](docs/en/exports-and-troubleshooting.md)
- [Documentación en español](docs/es/README.md)

## Privacy

LLM support is optional. When enabled, TrainLens minimizes and bounds notebook evidence,
redacts secrets, and keeps notebook-derived evidence separate from trusted system
instructions. Treat generated explanations as assistance and verify them against the
underlying training evidence.

## Scope

TrainLens is a lightweight notebook reporting and experiment-understanding layer, not a
full MLOps platform. It is designed for research and small-to-medium workflows where the
important training context already lives in Python and Jupyter.

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md) to contribute and [SECURITY.md](SECURITY.md) to
report vulnerabilities.

TrainLens is licensed under the [Apache License 2.0](LICENSE).
