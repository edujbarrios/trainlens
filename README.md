# TrainLens

TrainLens generates LLM-written training reports inside the Jupyter notebook you
are already using. It reads context from memory: model, dataset, metrics, logs,
traces, hyperparameters, and notes.

[![CI](https://github.com/edujbarrios/trainlens/actions/workflows/ci.yml/badge.svg)](https://github.com/edujbarrios/trainlens/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![PyPI](https://img.shields.io/pypi/v/trainlens.svg)](https://pypi.org/project/trainlens/)

<p align="center"><strong><em>Maintained by Eduardo J. Barrios.</em></strong></p>

It is built for research workflows with many training jobs, where you need
consistent explanations of results, datasets, and hyperparameters.

## How It Works

TrainLens reads the active notebook memory, extracts training context, redacts
likely secrets, and sends structured evidence to your configured
OpenAI-compatible LLM.

The LLM receives a TrainLens prompt that asks it to explain only what the
notebook evidence supports. It returns a Markdown report with the same shape
each time: summary, evidence, interpretation, risks, next steps, and bottom
line.

## Install

```bash
python -m pip install trainlens
```

Development install:

```bash
git clone https://github.com/edujbarrios/trainlens.git
cd trainlens
python -m pip install -e ".[dev]"
```

## Quickstart

```python
import os

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

%load_ext trainlens.magic.extension
%explain_training
```

## Example Output

For the Quickstart run above, TrainLens returns a structured report like this:

```markdown
## TrainLens Report

### Run summary
- Dataset context: `ag_news`
- Model context: `distilbert-base-uncased`
- Training params: `epochs=3`, `batch_size=32`, `learning_rate=5e-05`

### Evidence
- Training loss decreases: `0.62 -> 0.31 -> 0.18`
- Training accuracy increases: `0.78 -> 0.91 -> 0.96`
- Validation loss improves, then worsens: `0.48 -> 0.44 -> 0.57`
- Validation accuracy improves slightly, then slips: `0.84 -> 0.86 -> 0.85`

### Interpretation
- Optimization is working on the training set.
- Generalization peaks around epoch 2.
- Epoch 3 shows validation drift consistent with overfitting.

### Risks and caveats
- **Overfitting risk is supported by the metrics.**
- **Calibration/confidence drift is possible** because validation loss rises
  while validation accuracy changes only slightly.

### What to do next in the notebook
- Select epoch 2 as the current best checkpoint.
- Add early stopping on validation loss.
- Test 2 epochs and a slightly lower learning rate.

### Bottom line
- The run trained successfully, but the best generalization was reached at
  **epoch 2**, not epoch 3.
```

## Usage

TrainLens reads common notebook artifacts such as Keras histories, Hugging Face
`log_history`, PyTorch loop metrics, dataset notes, LoRA settings, trainable
parameter ratios, multimodal hints, and eval metrics.

```python
# PyTorch-style loop metrics kept in notebook memory.
dataset_name = "cifar10"
dataset_notes = "50k train images, 10 classes, CPU smoke-test subset."
hardware_notes = "Trained on CPU from a small in-memory DataLoader subset."
pytorch_loop_metrics = [
    {"epoch": 1, "train_loss": 1.42, "val_loss": 1.20, "val_accuracy": 0.58},
    {"epoch": 2, "train_loss": 0.94, "val_loss": 0.88, "val_accuracy": 0.69},
]

# Then ask TrainLens to explain the current notebook state with the LLM.
%load_ext trainlens.magic.extension
%explain_training

# After several runs in the same notebook, compare captured metric snapshots.
%compare_runs
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
