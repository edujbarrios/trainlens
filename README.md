# TrainLens

**Research-grade reports for AI training runs, directly from Python notebooks.**

TrainLens is a lightweight library for understanding, documenting, and comparing
model-training experiments. It reads the state already present in a notebook,
normalizes common metrics, detects useful evidence, redacts likely secrets, and
can use an OpenAI-compatible LLM to draft a scientific report or improvement
plan.

It is meant for research workflows where experiments move quickly and the
important context lives in Python variables, not in a separate dashboard.

<p align="center">
  <a href="https://github.com/edujbarrios/trainlens/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/edujbarrios/trainlens/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://pypi.org/project/trainlens/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/trainlens?logo=pypi"></a>
  <a href="pyproject.toml"><img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-blue"></a>
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-yellow"></a>
</p>

## Release Status

TrainLens `0.8.0` is the current release line. Package metadata, release notes,
documentation, tests, and PyPI distribution checks are aligned for this version.
The PyPI badge above reflects the latest version actually published to the
registry.

## What It Does

| Feature | Purpose |
| --- | --- |
| Notebook reports | Explain training metrics, signals, limitations, and next steps. |
| Framework adapters | Read Keras, Hugging Face, and Lightning training objects without hard dependencies. |
| Run comparison | Compare baseline and experiment metrics with improvement/regression labels. |
| Export | Write Markdown, HTML, JSON, and optional PDF artifacts. |
| Privacy guardrails | Redact likely secrets before LLM prompts are created. |
| Built-in prompts | Select and customize explanations for different training objectives. |
| Real-time monitoring | Stream metrics and detect training anomalies while a run is active. |
| Next-experiment planning | Turn run evidence into a controlled, measurable follow-up. |

TrainLens has two layers:

1. A local deterministic layer for metric extraction, framework detection,
   heuristics, comparison, and export.
2. An optional LLM layer for paper-style reports and experiment suggestions.

## Install

```bash
pip install trainlens
```

Optional PDF support:

```bash
pip install "trainlens[pdf]"
```

## Quickstart

```python
history = {
    "train_loss": [0.61, 0.50, 0.25],
    "eval_loss": [0.62, 0.52, 0.28],
    "accuracy": [0.89, 0.91, 0.92],
    "val_accuracy": [0.82, 0.83, 0.92],
}
```

```python
%load_ext trainlens.magic.extension
%explain_training
%suggest_improvements
%compare_runs
```

`%explain_training` captures a run. After two captured runs, `%compare_runs`
compares the latest run with the previous one.

To enable LLM reports, configure any OpenAI-compatible endpoint:

```python
import os

os.environ["TRAINLENS_LLM_BASE_URL"] = "https://api.openai.com/v1"
os.environ["TRAINLENS_LLM_API_KEY"] = "your-api-key"
os.environ["TRAINLENS_LLM_MODEL"] = "your-model"
```

Local endpoints such as Ollama, LM Studio, vLLM, and llama.cpp server work with
the same environment variables.

## Framework Adapters

TrainLens can inspect common training objects already present in a notebook:

```python
# Keras / TensorFlow
history = model.fit(x_train, y_train, validation_data=(x_val, y_val))

# Hugging Face
trainer.train()  # reads trainer.state.log_history

# PyTorch Lightning
trainer.fit(module, datamodule=datamodule)  # reads callback/logged metrics

# Plain PyTorch
# Keep model, optimizer, scheduler, and train_loader in the notebook namespace.
# TrainLens reads parameter counts, optimizer groups, current learning rates,
# scheduler epoch, batching settings, and dataset size without importing PyTorch.

%explain_training
```

Adapters use duck typing and module names. TrainLens does not require
TensorFlow, PyTorch, Transformers, or Lightning as package dependencies.

## Python API

Build and export notebook reports:

```python
from trainlens import build_paper_report, write_report

paper = build_paper_report(max_metric_points=12)
write_report(paper, "trainlens-report.html")
write_report(paper, "trainlens-report.json")
```

Long metric histories are summarized with their endpoints, extrema, observation
count, and an ordered sample. Adjust `max_metric_points` to trade prompt size for
curve detail; values below `2` are rejected so training endpoints are always kept.

Compare runs:

```python
from trainlens import compare_runs, render_run_comparison

comparison = compare_runs(
    {"validation_loss": 0.52, "validation_accuracy": 0.84},
    {"validation_loss": 0.47, "validation_accuracy": 0.87},
    baseline_name="baseline",
    experiment_name="lower learning rate",
)

print(render_run_comparison(comparison))
```

Export comparisons through the same report API:

```python
from trainlens import write_report

write_report(comparison, "comparison.html")
write_report(comparison, "comparison.json")
```

Metric direction is inferred from common names: loss-like metrics are better
when they decrease; accuracy, F1, recall, precision, AUC, and score-like metrics
are better when they increase. Unknown metrics are shown with deltas but without
an improvement/regression claim.

## Built-in prompts

TrainLens includes prompts for scientific reporting, improvement planning,
training diagnosis, and controlled experiment design. Discover the available
prompts and their intended use:

```python
from trainlens import show_trainlens_prompts

for prompt in show_trainlens_prompts():
    print(f"{prompt.name}: {prompt.description}")
```

Choose a built-in prompt and parameterize it for the goal of the analysis:

```python
from trainlens import PromptOptions, build_paper_report

options = PromptOptions(
    prompt_name="training_diagnosis",
    objective="Explain why validation loss rose after epoch 8.",
    model_family="vision transformer fine-tune",
    audience="computer-vision researchers",
    tone="concise, technical, and cautious",
    focus_areas=("overfitting", "learning-rate schedule", "augmentation"),
    rules=(
        "Use only evidence contained in the notebook context.",
        "Rank each hypothesis by confidence.",
    ),
    return_instructions=(
        "Return observations, ranked hypotheses, verification checks, and next actions.",
    ),
)

report = build_paper_report(globals(), prompt_options=options)
print(report.markdown)
```

The configurable fields are `prompt_name`, `objective`, `heading`,
`model_family`, `audience`, `tone`, `rules`, `focus_areas`, and
`return_instructions`. Use `get_trainlens_prompt(name)` to inspect one built-in
definition. Prompt construction still applies TrainLens secret redaction before
notebook context is sent to an LLM provider.

## Real-time monitoring

`TrainLensMonitor` processes metrics incrementally instead of waiting for a run
to finish. It currently detects non-finite values, stagnant losses, and possible
overfitting when training loss falls while validation loss rises. Every alert
contains a stable code, severity, step, message, and the evidence that triggered
it.

```python
from trainlens import MonitorConfig, TrainLensMonitor

monitor = TrainLensMonitor(MonitorConfig(patience=3, min_delta=0.01))

for epoch, metrics in enumerate(training_loop()):
    alerts = monitor.observe(epoch, metrics)
    for alert in alerts:
        print(alert.severity, alert.message, alert.evidence)
```

`TrainLensCallback` provides dependency-free hooks shaped for Keras,
Hugging Face Transformers, and PyTorch Lightning. It can collect alerts,
request a stop after a critical anomaly, and invoke an application-defined
handler periodically through `explain_every`.

```python
from trainlens import TrainLensCallback

def explain_snapshot(observation):
    print(f"Explain step {observation.step}: {dict(observation.metrics)}")

callback = TrainLensCallback(
    alerts=True,
    explain_every=5,
    on_explain=explain_snapshot,
    stop_on_anomaly=False,
)

# Keras-style hook
callback.on_epoch_end(epoch, logs)

# Transformers-style hook
callback.on_log(state=trainer_state, control=trainer_control, logs=logs)

# Lightning-style hook
callback.on_train_epoch_end(trainer)
```

Automatic stopping is opt-in and currently applies only to critical alerts,
such as a NaN or infinite metric. The monitoring engine is local and
deterministic; `explain_every` calls the supplied handler but does not contact
an LLM unless that handler explicitly does so.

## Next-experiment recommendations

TrainLens can turn completed runs into a structured proposal for the next
controlled experiment. Recommendations contain a hypothesis, one parameter
change, parameters to keep constant, measurable success criteria, estimated
cost, confidence, and the evidence used to make the proposal.

```python
from trainlens import (
    ExperimentRun,
    experiment_config,
    render_next_experiment,
    suggest_next_experiment,
)

runs = [
    ExperimentRun(
        name="baseline",
        metrics={"train_loss": 0.20, "validation_loss": 0.50},
        parameters={"learning_rate": 1e-3, "dropout": 0.10, "batch_size": 32},
        estimated_cost="medium",
    ),
]

recommendation = suggest_next_experiment(
    runs,
    objective_metric="validation_loss",
    minimum_improvement=0.01,
)

print(render_next_experiment(recommendation))

next_config = experiment_config(
    recommendation,
    base_parameters=runs[0].parameters,
)
```

The initial implementation is local and deterministic. It selects the strongest
source run for the objective, detects evidence such as a generalization gap, and
changes only one variable so that the result is easier to interpret. It does not
claim that a recommendation is guaranteed to improve the model; its confidence
and evidence fields are intended to make that uncertainty explicit.

## What TrainLens Inspects

TrainLens looks for lightweight evidence:

- metric histories in dictionaries, lists, logs, and trace-like events
- model-like objects with `fit`, `predict`, `score`, `state_dict`, or transformer config
- labels such as `y_train`, `labels`, or `target`
- metadata such as model names, dataset notes, and training parameters

Common aliases such as `eval_loss`, `val_loss`, `validation_loss`,
`train/accuracy`, and `eval-accuracy` are normalized automatically.

## Privacy

Local extraction, framework adapters, run comparison, and export do not contact
an external service. LLM calls happen only when you use an LLM report helper or
magic with a configured provider.

TrainLens redacts likely secrets and truncates large literals before prompt
construction. Do not store real API keys in notebooks or committed files.

## Development

```bash
pip install -e ".[dev]"
python -m pytest
python -m ruff check .
mypy src/trainlens
python -m build --no-isolation
python -m twine check dist/trainlens-0.6.0*
```

New framework support should stay optional and be tested with fake objects or
duck typing fixtures instead of adding heavy ML frameworks as required
dependencies.

## License

Apache License 2.0. See [LICENSE](LICENSE).
