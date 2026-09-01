# Installation and quickstart

## Install

```bash
python -m pip install trainlens
```

For PDF export:

```bash
python -m pip install "trainlens[pdf]"
```

For local development:

```bash
git clone https://github.com/edujbarrios/trainlens.git
cd trainlens
python -m pip install -e ".[dev]"
python -m pytest
```

## Five-minute notebook workflow

Create or train a model so that a metric history remains in the notebook:

```python
history = {
    "train_loss": [0.61, 0.50, 0.25],
    "eval_loss": [0.62, 0.52, 0.28],
    "accuracy": [0.89, 0.91, 0.92],
    "val_accuracy": [0.82, 0.83, 0.92],
}
```

Configure an LLM provider, then load the extension:

```python
%env TRAINLENS_LLM_BASE_URL=https://api.openai.com/v1
%env TRAINLENS_LLM_API_KEY=replace-me
%env TRAINLENS_LLM_MODEL=your-model
%load_ext trainlens.magic.extension
```

```python
%explain_training
%suggest_improvements
```

`%explain_training` also captures the metrics in an in-memory run store. Run it
after a second experiment and then use `%compare_runs`. The store belongs to the
loaded IPython extension and is not a persistent experiment database.

## First local-only workflow

Comparison, monitoring, experiment recommendations, and export do not need an
LLM:

```python
from trainlens import compare_runs, render_run_comparison

comparison = compare_runs(
    {"validation_loss": 0.52, "accuracy": 0.84},
    {"validation_loss": 0.47, "accuracy": 0.87},
    baseline_name="baseline",
    experiment_name="lower learning rate",
)
print(render_run_comparison(comparison))
```

## Next steps

Use [notebooks.md](notebooks.md) for automatic inspection, or
[python-api.md](python-api.md) when you want explicit, testable Python calls.

