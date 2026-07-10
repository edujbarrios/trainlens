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
  <a href="https://pypi.org/project/trainlens/"><img alt="TrainLens 0.6.0" src="https://img.shields.io/badge/trainlens-0.6.0-blue?logo=pypi"></a>
  <a href="pyproject.toml"><img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-blue"></a>
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-yellow"></a>
</p>

## What It Does

| Feature | Purpose |
| --- | --- |
| Notebook reports | Explain training metrics, signals, limitations, and next steps. |
| Framework adapters | Read Keras, Hugging Face, and Lightning training objects without hard dependencies. |
| Run comparison | Compare baseline and experiment metrics with improvement/regression labels. |
| Export | Write Markdown, HTML, JSON, and optional PDF artifacts. |
| Privacy guardrails | Redact likely secrets before LLM prompts are created. |

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

%explain_training
```

Adapters use duck typing and module names. TrainLens does not require
TensorFlow, PyTorch, Transformers, or Lightning as package dependencies.

## Python API

Build and export notebook reports:

```python
from trainlens import build_paper_report, write_report

paper = build_paper_report()
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
