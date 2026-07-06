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

## Documentation

Full documentation lives in [documentation](documentation/README.md), including
usage, configuration, supported evidence, provider setup, API reference, and the
release process.

## License

Apache License 2.0. See [LICENSE](LICENSE).
