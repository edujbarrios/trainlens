# TrainLens

**Research-grade reports for AI training runs, directly from your notebook.**

TrainLens is a lightweight Python library for turning model-training state into
structured, reproducible analysis. It inspects the variables already present in
a Jupyter notebook, normalizes common training metrics, detects useful evidence,
redacts likely secrets, and can draft a scientific report or experiment plan
with any OpenAI-compatible LLM provider.

It is built for practical research workflows: fine-tuning runs, model debugging,
small lab experiments, and repeated comparisons where the important context is
already in Python.

<p align="center">
  <a href="https://github.com/edujbarrios/trainlens/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/edujbarrios/trainlens/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://pypi.org/project/trainlens/"><img alt="TrainLens 0.6.0" src="https://img.shields.io/badge/trainlens-0.6.0-blue?logo=pypi"></a>
  <a href="pyproject.toml"><img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-blue"></a>
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-yellow"></a>
</p>

## Why TrainLens

TrainLens helps when you want to understand, document, or compare experiments
without manually copying notebook context into a chat window.

| Task | What TrainLens does |
| --- | --- |
| Explain a run | Builds evidence-backed notebook reports from metrics and model state. |
| Suggest next experiments | Generates grounded improvement plans with LLM provenance. |
| Compare runs | Classifies metric movement as improved, regressed, unchanged, new, or removed. |
| Export artifacts | Writes Markdown, HTML, JSON, and optional PDF reports. |
| Keep frameworks optional | Reads Keras, Hugging Face, and Lightning objects by duck typing. |
| Protect context | Redacts likely secrets before creating LLM prompts. |

TrainLens works in two layers. The deterministic layer extracts metrics,
framework evidence, and heuristic signals locally. The optional LLM layer uses
that compact evidence to write a paper-style report or improvement plan.

## Install

```bash
pip install trainlens
```

For PDF export:

```bash
pip install "trainlens[pdf]"
```

## Quickstart

Keep ordinary experiment state in your notebook:

```python
dataset_name = "synthetic_cpu_binary_classification"
model_name = "logistic-regression-baseline"
training_params = {"epochs": 30, "learning_rate": 0.45}

history = {
    "train_loss": [0.6095, 0.5005, 0.2459],
    "eval_loss": [0.6164, 0.5160, 0.2808],
    "accuracy": [0.8889, 0.9056, 0.9222],
    "val_accuracy": [0.8167, 0.8333, 0.9167],
}
```

Configure an OpenAI-compatible provider:

```python
import os

os.environ["TRAINLENS_LLM_BASE_URL"] = "https://api.openai.com/v1"
os.environ["TRAINLENS_LLM_API_KEY"] = "your-api-key"
os.environ["TRAINLENS_LLM_MODEL"] = "gpt-5.4-mini"
```

Use the notebook magics:

```python
%load_ext trainlens.magic.extension

%explain_training
%suggest_improvements
%compare_runs
```

`%explain_training` captures a run in the notebook-local store. Once at least
two runs are captured, `%compare_runs` compares the latest run against the
previous one.

## What Gets Inspected

TrainLens looks for lightweight evidence in the active namespace:

- metric histories in dictionaries, lists, framework logs, and trace-like events
- model-like objects with `fit`, `predict`, `score`, `state_dict`, or transformer config
- dataset labels such as `y_train`, `labels`, or `target`
- metadata such as model names, dataset notes, and training parameters

Metric names are normalized across common aliases, including `eval_loss`,
`val_loss`, `validation_loss`, `train/accuracy`, and `eval-accuracy`.

## Framework Adapters

TrainLens can read common training objects without requiring their frameworks as
dependencies.

### Keras / TensorFlow

```python
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=8,
)

%explain_training
```

TrainLens reads `history.history`, including keys such as `loss`, `accuracy`,
`val_loss`, and `val_accuracy`.

### Hugging Face Trainer

```python
trainer.train()

%explain_training
```

TrainLens reads `trainer.state.log_history` and extracts values such as `loss`,
`eval_loss`, `eval_accuracy`, `epoch`, and `global_step`.

### PyTorch Lightning

```python
trainer.fit(module, datamodule=datamodule)

%explain_training
```

TrainLens reads `callback_metrics`, `logged_metrics`, and
`progress_bar_metrics`. Tensor-like scalar values are converted through
`.item()` when available.

## Python API

Build notebook reports:

```python
from trainlens import build_improvement_ideas, build_paper_report, write_report

paper = build_paper_report()
ideas = build_improvement_ideas()

write_report(paper, "trainlens-report.md")
write_report(paper, "trainlens-report.html")
write_report(paper, "trainlens-report.json")
```

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

Use TrainLens results directly:

```python
from trainlens import compare_runs, write_report
from trainlens.pipeline import explain_namespace

baseline = explain_namespace({"history": baseline_history})
experiment = explain_namespace({"history": experiment_history})

comparison = compare_runs(baseline, experiment)
write_report(comparison, "run-comparison.html")
write_report(comparison, "run-comparison.json")
```

Metric direction is inferred from common names. Loss-like metrics are better
when they go down; accuracy, F1, recall, precision, AUC, and score-like metrics
are better when they go up. Unknown metrics are still shown with deltas, but
TrainLens does not claim whether they improved or regressed.

## Export

Reports and comparisons use the same export helpers:

```python
from trainlens import render_report, write_report

markdown = render_report(paper, format="markdown")
html = render_report(paper, format="html")
json_payload = render_report(paper, format="json")

write_report(paper, "report.md")
write_report(paper, "report.html")
write_report(paper, "report.json")
write_report(comparison, "comparison.html")
```

PDF export is optional:

```python
write_report(paper, "report.pdf")
```

## Local Models

TrainLens works with local OpenAI-compatible servers such as Ollama, LM Studio,
vLLM, and llama.cpp server.

Example with Ollama:

```bash
ollama pull llama3.1
ollama serve
```

```python
import os

os.environ["TRAINLENS_LLM_BASE_URL"] = "http://localhost:11434/v1"
os.environ["TRAINLENS_LLM_API_KEY"] = "ollama"
os.environ["TRAINLENS_LLM_MODEL"] = "llama3.1"
```

## Privacy

TrainLens summarizes visible notebook state, redacts likely secrets, truncates
large literals, and sends only the resulting compact context to the configured
LLM endpoint. Local metric extraction, framework adapters, run comparison, and
export helpers run without contacting an external service.

Do not store real API keys in notebooks or committed files.

## Development

```bash
pip install -e ".[dev]"
python -m pytest
python -m ruff check .
mypy src/trainlens
python -m build --no-isolation
python -m twine check dist/trainlens-0.6.0*
```

TrainLens intentionally keeps framework support optional. New adapters should
use fake test objects or duck typing tests instead of adding TensorFlow,
PyTorch, Transformers, or Lightning as required dependencies.

## License

Apache License 2.0. See [LICENSE](LICENSE).
