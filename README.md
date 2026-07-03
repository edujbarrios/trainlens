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

The generated report keeps a stable shape: summary, evidence, interpretation,
risks, next steps, and bottom line.

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

## Documentation

Full documentation lives in [documentation](documentation/README.md), including:

- why TrainLens exists
- notebook workflow
- supported evidence
- LLM provider configuration
- report interpretation
- troubleshooting
- API reference

## License

Apache License 2.0. See [LICENSE](LICENSE).
