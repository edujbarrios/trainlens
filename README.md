# TrainLens

TrainLens lets you generate LLM-written training reports without leaving the
Jupyter notebook you are running. It uses the context already in memory: model,
dataset, metrics, logs, traces, hyperparameters, and notes.

[![CI](https://github.com/edujbarrios/trainlens/actions/workflows/ci.yml/badge.svg)](https://github.com/edujbarrios/trainlens/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![PyPI](https://img.shields.io/pypi/v/trainlens.svg)](https://pypi.org/project/trainlens/)

Maintained by Eduardo J. Barrios.

It sends sanitized notebook training context to an OpenAI-compatible LLM and
displays a Markdown diagnosis in-place. A local heuristic report is still
available for debugging, but the main workflow is LLM-generated and requires a
configured provider.

## Install

Install the package from PyPI:

```bash
python -m pip install trainlens
```

For development, clone the repository and install it in editable mode:

```bash
git clone https://github.com/edujbarrios/trainlens.git
cd trainlens
python -m pip install -e .
```

For development tools and tests, install the `dev` extras:

```bash
python -m pip install -e ".[dev]"
```

## Quickstart

Set your provider details in the notebook or in the environment before calling
TrainLens:

```python
import os

from trainlens.notebook import display_llm_report

os.environ["TRAINLENS_LLM_BASE_URL"] = "https://api.openai.com/v1"
os.environ["TRAINLENS_LLM_API_KEY"] = "your-api-key"
os.environ["TRAINLENS_LLM_MODEL"] = "gpt-4.1-mini"
os.environ["TRAINLENS_LLM_TIMEOUT_SECONDS"] = "120"

dataset_name = "ag_news"
dataset_notes = "120k news titles; 4 classes; validation is balanced."
model_name = "distilbert-base-uncased"
training_params = {
    "epochs": 3,
    "batch_size": 32,
    "learning_rate": 5e-5,
    "max_length": 128,
}
history = {
    "train_loss": [0.62, 0.31, 0.18],
    "eval_loss": [0.48, 0.44, 0.57],
    "accuracy": [0.78, 0.91, 0.96],
    "val_accuracy": [0.84, 0.86, 0.85],
}

display_llm_report(globals())
```

TrainLens sends structured notebook context to the configured OpenAI-compatible
LLM. For this `ag_news` run, the generated report should connect the falling
training loss and rising validation loss after epoch 2 with likely overfitting:
the model is memorizing training headlines faster than it improves
generalization. The report then suggests next experiments such as stopping after
epoch 2, lowering the learning rate, or adding regularization.

Use any OpenAI-compatible provider by changing `TRAINLENS_LLM_BASE_URL`,
`TRAINLENS_LLM_API_KEY`, `TRAINLENS_LLM_MODEL`, and optionally
`TRAINLENS_LLM_TIMEOUT_SECONDS` for slower models.

## What It Answers

- What did this notebook train?
- Which dataset and training context shaped the result?
- Are the metrics improving, plateauing, overfitting, or missing evidence?
- What should I try next in this notebook?

TrainLens reads common notebook artifacts such as Keras histories, Hugging Face
`log_history`, PyTorch loop metrics, execution traces, dataset notes, LoRA
settings, trainable parameter ratios, multimodal hints, and eval metrics.

## Notebook Usage

Use the LLM helper or magic for the main workflow:

```python
from trainlens.notebook import display_llm_report

display_llm_report(globals())

%load_ext trainlens.magic.extension
%explain_training --llm
%compare_runs
```

For local debugging without a provider, use `display_live_report(globals())`.
That path is deterministic and heuristic-based; it is not the main LLM report.

## How It Works

```text
notebook variables
  -> namespace snapshot
  -> sanitized notebook context
  -> LLM report prompt
  -> Markdown report
```

Core modules: `introspection`, `analyzers`, `heuristics`, `models`, `llm`,
`magic`, `renderers`, and `storage`.

## Examples And Docs

- [notebooks/](notebooks/)
- [docs/live-notebook-cells.md](docs/live-notebook-cells.md)
- [docs/report-sections.md](docs/report-sections.md)

```bash
python examples/pytorch_loop_metrics.py
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
