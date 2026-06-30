# TrainLens

TrainLens lets you explain training results without leaving the Jupyter
notebook you are running. It uses the context already in memory: model, dataset,
metrics, logs, traces, hyperparameters, and notes.

[![CI](https://github.com/edujbarrios/trainlens/actions/workflows/ci.yml/badge.svg)](https://github.com/edujbarrios/trainlens/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![PyPI release](https://img.shields.io/badge/PyPI-planned%20after%20minor%20fixes-orange)](https://pypi.org/project/trainlens/)

Maintained by Eduardo J. Barrios.

It produces a local Markdown diagnosis plus an atlas-style notebook UI, and can
enhance that same report with an OpenAI-compatible LLM provider in-place.

## Quickstart

```python
import os

from IPython.display import Markdown, display
from trainlens.llm.enhancer import maybe_enhance
from trainlens.notebook import display_live_report


history = {"train_loss": [2.4, 1.9, 1.6], "eval_loss": [2.5, 2.0, 1.8]}
dataset_name = "customer-support-instructions"
dataset_notes = "Small validation split; long answers are underrepresented."
model_name = "mistral-lora-support-bot"
learning_rate = 2e-5

report = display_live_report(globals())

os.environ["TRAINLENS_LLM_BASE_URL"] = "https://api.example.com/v1"
os.environ["TRAINLENS_LLM_API_KEY"] = "your-api-key"
os.environ["TRAINLENS_LLM_MODEL"] = "your-model-name"

display(Markdown(maybe_enhance(report.markdown)))
```

The local atlas and report work without an API key. The LLM call is only used
for the enhanced explanation.

## What It Answers

- What did this notebook train?
- Which dataset and training context shaped the result?
- Are the metrics improving, plateauing, overfitting, or missing evidence?
- What should I try next in this notebook?

TrainLens reads common notebook artifacts such as Keras histories, Hugging Face
`log_history`, PyTorch loop metrics, execution traces, dataset notes, LoRA
settings, trainable parameter ratios, multimodal hints, and eval metrics.

## Notebook Usage

Load from a clone:

```python
from pathlib import Path
import sys

TRAINLENS_REPO = Path("/path/to/trainlens").resolve()
sys.path.insert(0, str(TRAINLENS_REPO / "src"))
```

Use the helper or magics:

```python
from trainlens.notebook import display_live_report, display_training_atlas

display_live_report(globals())
display_training_atlas(globals())

%load_ext trainlens.magic.extension
%explain_training
%training_atlas
%compare_runs
```

## How It Works

```text
notebook variables
  -> namespace snapshot
  -> model, dataset, metric, and trace extraction
  -> training heuristics
  -> Markdown report and atlas UI
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
