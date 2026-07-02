# TrainLens

TrainLens lets you explain training results without leaving the Jupyter
notebook you are running. It uses the context already in memory: model, dataset,
metrics, logs, traces, hyperparameters, and notes.

[![CI](https://github.com/edujbarrios/trainlens/actions/workflows/ci.yml/badge.svg)](https://github.com/edujbarrios/trainlens/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![PyPI release](https://img.shields.io/badge/PyPI-planned%20after%20minor%20fixes-orange)](https://pypi.org/project/trainlens/)

Maintained by Eduardo J. Barrios.

It produces a local Markdown diagnosis in the notebook output, and can enhance
that same report with an OpenAI-compatible LLM provider in-place.

## Install From Source

TrainLens is not published to PyPI yet. For now, clone the repository and install
it in editable mode:

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

```python
from trainlens.notebook import display_live_report

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

display_live_report(globals())
```

TrainLens explains that training loss keeps falling while validation loss rises
after epoch 2. For this `ag_news` run, that points to overfitting: the model is
memorizing training headlines faster than it improves generalization. The report
then suggests next experiments such as stopping after epoch 2, lowering the
learning rate, or adding regularization.

The local report works without an API key. `maybe_enhance` only calls an
OpenAI-compatible provider when `TRAINLENS_LLM_BASE_URL`,
`TRAINLENS_LLM_API_KEY`, and `TRAINLENS_LLM_MODEL` are configured.

## What It Answers

- What did this notebook train?
- Which dataset and training context shaped the result?
- Are the metrics improving, plateauing, overfitting, or missing evidence?
- What should I try next in this notebook?

TrainLens reads common notebook artifacts such as Keras histories, Hugging Face
`log_history`, PyTorch loop metrics, execution traces, dataset notes, LoRA
settings, trainable parameter ratios, multimodal hints, and eval metrics.

## Notebook Usage

Use the helper or magics:

```python
from trainlens.notebook import display_live_report

display_live_report(globals())

%load_ext trainlens.magic.extension
%explain_training
%compare_runs
```

## How It Works

```text
notebook variables
  -> namespace snapshot
  -> model, dataset, metric, and trace extraction
  -> training heuristics
  -> Markdown report
  -> optional LLM enhancement
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
