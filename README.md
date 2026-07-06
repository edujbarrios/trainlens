# TrainLens

TrainLens generates LLM-written training reports inside the Jupyter notebook you
are already using. It reads context from memory: model, dataset, metrics, logs,
traces, hyperparameters, and notes.

<p align="center">
  <a href="https://github.com/edujbarrios/trainlens/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/edujbarrios/trainlens/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://pypi.org/project/trainlens/"><img alt="PyPI" src="https://img.shields.io/pypi/v/trainlens?label=PyPI&logo=pypi"></a>
  <a href="pyproject.toml"><img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-blue"></a>
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-yellow"></a>
</p>

<p align="center"><strong><em>Maintained by Eduardo J. Barrios.</em></strong></p>

TrainLens is built for research workflows with many training jobs, where you
need consistent explanations of results, datasets, and hyperparameters without
copying notebook state into a separate prompt by hand.

## Install

```bash
pip install trainlens
```

Upgrade to the latest public version:

```bash
python -m pip install --upgrade trainlens
```

Development install:

```bash
git clone https://github.com/edujbarrios/trainlens.git
cd trainlens
python -m pip install -e ".[dev]"
```

## Report Modes

TrainLens has two LLM-backed notebook modes:

- `%explain_training` creates a scientific paper-style report with results,
  discussion, possible conclusions, limitations, and LLM provenance.
- `%suggest_improvements` creates a separate improvement plan with prioritized
  follow-up experiments.

Python helpers expose the same flows:

```python
from trainlens import build_improvement_ideas, build_paper_report

paper = build_paper_report(globals())
ideas = build_improvement_ideas(globals())
```

## Quickstart

```python
import os

os.environ["TRAINLENS_LLM_BASE_URL"] = "https://api.openai.com/v1"
os.environ["TRAINLENS_LLM_API_KEY"] = "your-api-key"
os.environ["TRAINLENS_LLM_MODEL"] = "gpt-4.1-mini"

dataset_name = "ag_news"
dataset_notes = "120k news titles; 4 classes; validation is balanced."
model_name = "distilbert-base-uncased"
training_params = {"epochs": 3, "batch_size": 32, "learning_rate": 5e-5}
history = {
    "train_loss": [0.62, 0.31, 0.18],
    "eval_loss": [0.48, 0.44, 0.57],
    "accuracy": [0.78, 0.91, 0.96],
    "val_accuracy": [0.84, 0.86, 0.85],
}

%load_ext trainlens.magic.extension
%explain_training
%suggest_improvements
```

## Reproducible CPU Example

The notebook in `examples/cpu_training_with_trainlens.ipynb` runs a tiny
CPU-only logistic-regression smoke test with pure Python. With the fixed seed
from that notebook, the run trains on 180 synthetic samples, validates on 60,
and produces the following final metrics:

```text
train_loss: 0.6095 -> 0.2459
eval_loss:  0.6164 -> 0.2808
accuracy:   0.8889 -> 0.9222
val_accuracy: 0.8167 -> 0.9167
```

Here is the minimal notebook shape:

```python
import math
import os
import random

os.environ["TRAINLENS_LLM_BASE_URL"] = "https://api.openai.com/v1"
os.environ["TRAINLENS_LLM_API_KEY"] = "your-api-key"
os.environ["TRAINLENS_LLM_MODEL"] = "gpt-4.1-mini"

random.seed(7)

dataset_name = "synthetic_cpu_binary_classification"
dataset_notes = (
    "240 synthetic 2D samples; binary labels; validation split is balanced; "
    "intended as a CPU-only smoke test."
)
model_name = "pure-python-logistic-regression"
training_params = {
    "epochs": 30,
    "learning_rate": 0.45,
    "train_samples": 180,
    "validation_samples": 60,
    "optimizer": "manual gradient descent",
}

history = {
    "train_loss": [0.6095, 0.5474, 0.5005, 0.4641, 0.4351, 0.4115, 0.3919,
                   0.3754, 0.3612, 0.3489, 0.3381, 0.3285, 0.32, 0.3123,
                   0.3053, 0.299, 0.2932, 0.2879, 0.2829, 0.2784, 0.2741,
                   0.2702, 0.2665, 0.263, 0.2597, 0.2566, 0.2537, 0.251,
                   0.2484, 0.2459],
    "eval_loss": [0.6164, 0.5593, 0.516, 0.4824, 0.4557, 0.4339, 0.4158,
                  0.4005, 0.3874, 0.376, 0.366, 0.3572, 0.3493, 0.3422,
                  0.3357, 0.3299, 0.3245, 0.3195, 0.315, 0.3108, 0.3068,
                  0.3032, 0.2998, 0.2965, 0.2935, 0.2907, 0.288, 0.2855,
                  0.2831, 0.2808],
    "accuracy": [0.8889, 0.9, 0.9056, 0.9167, 0.9167, 0.9167, 0.9167, 0.9167,
                 0.9167, 0.9167, 0.9167, 0.9167, 0.9167, 0.9111, 0.9111,
                 0.9111, 0.9111, 0.9111, 0.9111, 0.9111, 0.9111, 0.9111,
                 0.9111, 0.9111, 0.9111, 0.9111, 0.9111, 0.9111, 0.9167,
                 0.9222],
    "val_accuracy": [0.8167, 0.8333, 0.8333, 0.8333, 0.8333, 0.8333, 0.8333,
                     0.85, 0.85, 0.8667, 0.8833, 0.8833, 0.8833, 0.9, 0.9,
                     0.9, 0.9, 0.9, 0.9, 0.9, 0.8833, 0.8833, 0.8833,
                     0.8833, 0.8833, 0.9, 0.9, 0.9, 0.9167, 0.9167],
}

final_metrics = {
    "train_loss": history["train_loss"][-1],
    "validation_loss": history["eval_loss"][-1],
    "train_accuracy": history["accuracy"][-1],
    "validation_accuracy": history["val_accuracy"][-1],
}

trace_log = [
    {
        "epoch": epoch,
        "train_loss": history["train_loss"][epoch - 1],
        "eval_loss": history["eval_loss"][epoch - 1],
        "accuracy": history["accuracy"][epoch - 1],
        "val_accuracy": history["val_accuracy"][epoch - 1],
    }
    for epoch in range(1, training_params["epochs"] + 1)
]

%load_ext trainlens.magic.extension
%explain_training
%suggest_improvements
```

TrainLens turns that notebook state into a report noting that both training and
validation loss decrease monotonically, the final train/validation gap is small,
and there is no visible validation drift in the 30-epoch window.

## How It Works

When a notebook magic runs, TrainLens:

1. reads the active IPython namespace
2. summarizes visible training context
3. redacts likely secrets
4. sends evidence to an OpenAI-compatible chat completions endpoint
5. displays Markdown in the notebook

The prompt tells the LLM to use only supplied notebook evidence and to name the
configured LLM model in the report provenance section.

## Example Output

`%explain_training` starts like this:

```markdown
## TrainLens Scientific Report

### LLM provenance
- Drafted with `gpt-4.1-mini` through an OpenAI-compatible provider.

### Abstract
The run successfully optimized the training objective, with training loss
falling from `0.62` to `0.18`. Validation evidence is weaker: validation loss
improved through epoch 2 and then worsened at epoch 3.

### Results
- Training loss decreases: `0.62 -> 0.31 -> 0.18`
- Validation loss improves, then worsens: `0.48 -> 0.44 -> 0.57`
```

`%suggest_improvements` starts like this:

```markdown
## TrainLens Improvement Ideas

### Evidence Snapshot
- Validation loss worsens after epoch 2.
- Validation accuracy is nearly flat between epochs 2 and 3.

### Prioritized Experiments
1. Select the epoch-2 checkpoint and compare it against the final checkpoint.
2. Add early stopping on validation loss.
3. Try a slightly lower learning rate or fewer epochs.
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
