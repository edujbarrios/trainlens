# TrainLens

**Turn AI training runs into research-grade notebook reports.**

TrainLens is a Jupyter-first support package for AI model training and research
workflows. Its main value is that you can turn the training state already
present in your notebook into structured analysis without leaving Jupyter,
opening a separate chat, or copying experiment context by hand.

It redacts likely secrets and asks an OpenAI-compatible LLM to draft:

- a scientific paper-style training report
- an evidence-backed improvement plan

It is useful when you run many experiments and want consistent explanations of
metrics, datasets, hyperparameters, limitations, and next steps directly inside
the notebook where the work is happening.

## What TrainLens Gives You

TrainLens turns notebook state into evidence-backed reports for model training.
It is designed for research notebooks, fine-tuning experiments, small lab
projects, and practical ML debugging where the important context already lives
in Python variables.

| Need | TrainLens support |
| --- | --- |
| Explain a training run | Notebook magics and Python helpers generate structured reports. |
| Avoid copying context into chat | TrainLens inspects visible notebook variables and builds compact evidence. |
| Keep reports reproducible | Export Markdown, HTML, JSON, and optional PDF artifacts. |
| Compare experiments | `compare_runs` classifies metric improvements, regressions, and missing metrics. |
| Use common training frameworks | Lightweight adapters read Keras, Hugging Face, and Lightning objects. |
| Protect sensitive context | Likely secrets are redacted before LLM prompts are created. |

TrainLens can be useful with or without an LLM. The deterministic analysis
extracts metrics, framework evidence, and local heuristic signals. When an
OpenAI-compatible provider is configured, TrainLens can also ask the model to
draft a paper-style report or an improvement plan grounded in the same evidence.

<p align="center">
  <a href="https://github.com/edujbarrios/trainlens/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/edujbarrios/trainlens/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://pypi.org/project/trainlens/"><img alt="TrainLens 0.6.0" src="https://img.shields.io/badge/trainlens-0.6.0-blue?logo=pypi"></a>
  <a href="pyproject.toml"><img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-blue"></a>
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-yellow"></a>
</p>

<p align="center">
  <strong>Actively maintained by Eduardo J. Barrios.</strong><br>
  For issues or improvements, open a pull request or email
  <a href="mailto:edujbarrios@outlook.com">edujbarrios@outlook.com</a>.
</p>

## Quickstart

Install TrainLens:

```bash
pip install trainlens
```

Configure an OpenAI-compatible LLM provider:

```python
import os

os.environ["TRAINLENS_LLM_BASE_URL"] = "https://api.openai.com/v1"
os.environ["TRAINLENS_LLM_API_KEY"] = "your-api-key"
os.environ["TRAINLENS_LLM_MODEL"] = "gpt-5.4-mini"
```

Keep useful experiment state in ordinary notebook variables:

```python
dataset_name = "synthetic_cpu_binary_classification"
dataset_notes = "240 synthetic 2D samples; balanced validation split."
model_name = "pure-python-logistic-regression"
training_params = {"epochs": 30, "learning_rate": 0.45}

history = {
    "train_loss": [0.6095, 0.5005, 0.2459],
    "eval_loss": [0.6164, 0.5160, 0.2808],
    "accuracy": [0.8889, 0.9056, 0.9222],
    "val_accuracy": [0.8167, 0.8333, 0.9167],
}
```

Run the notebook magics:

```python
%load_ext trainlens.magic.extension

# Scientific paper-style training report
%explain_training

# Evidence-backed experiment plan
%suggest_improvements
```

| Magic | Output |
| --- | --- |
| `%explain_training` | Paper-style report with results, discussion, limitations, and LLM provenance. |
| `%suggest_improvements` | Follow-up experiment plan grounded in the same notebook evidence. |

## Local Models

TrainLens can avoid external API token costs by using a local OpenAI-compatible
server. Example with Ollama:

```bash
ollama pull llama3.1
ollama serve
```

```python
import os

os.environ["TRAINLENS_LLM_BASE_URL"] = "http://localhost:11434/v1"
os.environ["TRAINLENS_LLM_API_KEY"] = "ollama"
os.environ["TRAINLENS_LLM_MODEL"] = "llama3.1"
```

The same pattern works with LM Studio, vLLM, llama.cpp server, and similar
local servers.

## Framework Adapters

TrainLens can read common training objects already present in a notebook:

- Keras / TensorFlow `History` objects from `model.fit(...)`
- Hugging Face `Trainer` objects with `state.log_history`
- PyTorch Lightning `Trainer` objects with callback or logged metrics

These adapters use duck typing, so the heavy ML frameworks remain optional.
When one of these objects is found, `%explain_training` merges its metrics into
the normal TrainLens report automatically.

### Keras / TensorFlow

```python
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=8,
)

%explain_training
```

TrainLens reads `history.history`, including keys such as `loss`, `accuracy`,
`val_loss`, and `val_accuracy`.

### Hugging Face Trainer

```python
trainer.train()

# trainer.state.log_history is inspected automatically
%explain_training
```

TrainLens looks for `Trainer`-style logs such as `loss`, `eval_loss`,
`eval_accuracy`, `epoch`, and `global_step`. If the trainer exposes a model,
TrainLens also uses that model class as framework evidence.

### PyTorch Lightning

```python
trainer.fit(module, datamodule=datamodule)

# callback_metrics, logged_metrics, and progress_bar_metrics are supported
%explain_training
```

Tensor-like scalar values are converted through `.item()` when available, so
common Lightning metric objects can be read without importing PyTorch directly.

### Manual Metrics Still Work

You can always provide simple dictionaries or lists:

```python
history = {
    "train_loss": [0.82, 0.61, 0.49],
    "validation_loss": [0.88, 0.69, 0.57],
    "train_accuracy": [0.71, 0.79, 0.84],
    "validation_accuracy": [0.68, 0.74, 0.8],
}

%explain_training
```

## Python API

```python
from trainlens import build_improvement_ideas, build_paper_report, write_report

paper = build_paper_report()
ideas = build_improvement_ideas()

write_report(paper, "trainlens-report.md")
write_report(paper, "trainlens-report.html")
write_report(paper, "trainlens-report.json")
```

Inside Jupyter, helpers read the active notebook namespace automatically.
Outside IPython, pass an explicit dictionary-like namespace.

Compare two runs:

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

## Report Export

TrainLens reports can be exported as Markdown, HTML, JSON, and optionally PDF:

```python
from trainlens import render_report, write_report

markdown = render_report(paper, format="markdown")
html = render_report(paper, format="html")
json_payload = render_report(paper, format="json")

write_report(paper, "report.md")
write_report(paper, "report.html")
write_report(paper, "report.json")
```

PDF export uses an optional dependency:

```bash
pip install "trainlens[pdf]"
```

```python
write_report(paper, "report.pdf")
```

## Token Usage

Costs depend on provider, model, tokenizer, prompt size, and output length.
These are rough estimates, not billing records.

| Call | Input Tokens | Output Tokens | Total |
| --- | ---: | ---: | ---: |
| `%explain_training` paper report | 1K-2K | 2.5K-3.5K | 3.5K-5.5K |
| `%suggest_improvements` plan | 1K-2K | 2K-3K | 3K-5K |
| Both calls | 2K-4K | 4.5K-6.5K | 6.5K-10.5K |

Approximate USD cost for a 50% input / 50% output token mix, using public prices
checked on 2026-07-07:

| Provider / model | 1K tokens | 100K tokens | 1M tokens | 10M tokens |
| --- | ---: | ---: | ---: | ---: |
| OpenAI `gpt-5.4-mini` | $0.0026 | $0.2625 | $2.6250 | $26.2500 |
| Anthropic Claude Haiku 4.5 | $0.0030 | $0.3000 | $3.0000 | $30.0000 |
| Google Gemini 2.5 Flash-Lite | $0.0003 | $0.0250 | $0.2500 | $2.5000 |
| Local Ollama / LM Studio | $0 API cost | $0 API cost | $0 API cost | $0 API cost |

Cost formula:

```math
\text{estimated cost} =
\frac{
\text{input tokens} \times \text{input price per 1M}
+ \text{output tokens} \times \text{output price per 1M}
}{1{,}000{,}000}
```

## Privacy

TrainLens summarizes visible notebook state, redacts likely secrets, truncates
large literals, and sends only the resulting context to the configured endpoint.
Do not store real API keys in notebooks or committed files.

## License

Apache License 2.0. See [LICENSE](LICENSE).
