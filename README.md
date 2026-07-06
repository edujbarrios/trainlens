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

It is built for research workflows with many training jobs, where you need
consistent explanations of results, datasets, and hyperparameters.

## How It Works

TrainLens is a small notebook pipeline. Conceptually, `%explain_training` does
this:

```python
%explain_training

# trainlens.magic.commands
# Reads the active IPython namespace from the current notebook.
namespace = get_ipython().user_ns

# trainlens.llm.context + trainlens.introspection
# Finds visible training context: model names, histories, metric logs, dataset
# notes, hyperparameters, traces, and PEFT metadata.
context = build_llm_notebook_context(namespace)

# trainlens.security
# Redacts likely secrets while values are summarized and before prompt rendering.
safe_evidence = context.markdown

# trainlens.llm.prompts
# Wraps the evidence in TrainLens' internal report-generation prompt. The prompt
# tells the LLM to explain only what the notebook supports.
prompt = render_ml_results_explanation_prompt(safe_evidence)

# trainlens.llm.openai_compatible
# Sends the evidence to the configured chat-completions endpoint; the provider
# renders the prompt and returns Markdown for display in the notebook.
report = OpenAICompatibleProvider(config).explain(safe_evidence)
display(Markdown(report))
```

The generated report is prompted as a scientific-style Markdown report with
results, interpretation, possible conclusions, limitations, and LLM provenance.
You can also run a separate improvement-ideas mode for follow-up experiments.

## Install

```bash
pip install trainlens
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

import trainlens

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
%suggest_improvements
```

## Example Output

For the Quickstart run above, `%explain_training` returns a scientific-style
report like this:

```markdown
## TrainLens Scientific Report

### LLM provenance
- Drafted with `gpt-4.1-mini` through an OpenAI-compatible provider.

### Abstract
The run successfully optimized the training objective, with training loss
falling from `0.62` to `0.18`. Validation evidence is weaker: validation loss
improved through epoch 2 and then worsened at epoch 3.

### Methods Context
- Dataset context: `ag_news`
- Model context: `distilbert-base-uncased`
- Training params: `epochs=3`, `batch_size=32`, `learning_rate=5e-05`

### Results
- Training loss decreases: `0.62 -> 0.31 -> 0.18`
- Training accuracy increases: `0.78 -> 0.91 -> 0.96`
- Validation loss improves, then worsens: `0.48 -> 0.44 -> 0.57`
- Validation accuracy improves slightly, then slips: `0.84 -> 0.86 -> 0.85`

### Discussion
The metrics suggest that optimization continued on the training set after the
best validation-loss point. The epoch-3 validation-loss increase is consistent
with overfitting or calibration drift, especially because validation accuracy
does not improve alongside the lower training loss.

### Possible Conclusions
The best generalization evidence appears around epoch 2 rather than epoch 3.
The run is useful, but the final checkpoint should not be assumed to be the best
checkpoint without further validation.

### Limitations
The report only uses notebook evidence. It cannot verify held-out test
performance, dataset leakage, or class-level behavior unless those values are
available in memory.
```

`%suggest_improvements` produces a separate improvement plan:

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

Full documentation lives in [documentation](documentation/README.md), including:

- why TrainLens exists
- notebook workflow
- supported evidence
- LLM provider configuration
- report interpretation
- troubleshooting
- API reference
- release process

## License

Apache License 2.0. See [LICENSE](LICENSE).
