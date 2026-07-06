# TrainLens

TrainLens writes LLM-generated training reports directly inside Jupyter. It
reads the variables already in your notebook, such as dataset notes, model
names, hyperparameters, metric history, final metrics, and trace logs, then
sends a compact summary to an OpenAI-compatible chat completions endpoint.

<p align="center">
  <a href="https://github.com/edujbarrios/trainlens/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/edujbarrios/trainlens/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://pypi.org/project/trainlens/"><img alt="PyPI" src="https://img.shields.io/pypi/v/trainlens?label=PyPI&logo=pypi"></a>
  <a href="pyproject.toml"><img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-blue"></a>
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-yellow"></a>
</p>

<p align="center">
  <strong>Actively maintained by Eduardo J. Barrios.</strong><br>
  For improvements or issues, open a pull request or email
  <a href="mailto:edujbarrios@outlook.com">edujbarrios@outlook.com</a>.
</p>

## Install

```bash
pip install trainlens
```

## Configure An LLM

TrainLens is designed for OpenAI-compatible APIs. Set these variables before
running the notebook magics:

```python
import os

os.environ["TRAINLENS_LLM_BASE_URL"] = "https://api.openai.com/v1"
os.environ["TRAINLENS_LLM_API_KEY"] = "your-api-key"
os.environ["TRAINLENS_LLM_MODEL"] = "gpt-4.1-mini"
```

To avoid external API token costs, run a local model behind an
OpenAI-compatible server. For example, with Ollama:

```bash
ollama pull llama3.1
ollama serve
```

Then point TrainLens at Ollama's local OpenAI-compatible endpoint:

```python
import os

os.environ["TRAINLENS_LLM_BASE_URL"] = "http://localhost:11434/v1"
os.environ["TRAINLENS_LLM_API_KEY"] = "ollama"
os.environ["TRAINLENS_LLM_MODEL"] = "llama3.1"
```

The same pattern works with other local servers such as LM Studio, vLLM, or
llama.cpp server.

## Notebook Example

Keep useful training state in normal Python variables:

```python
dataset_name = "synthetic_cpu_binary_classification"
dataset_notes = "240 synthetic 2D samples; balanced validation split."
model_name = "pure-python-logistic-regression"
training_params = {"epochs": 30, "learning_rate": 0.45, "optimizer": "manual GD"}

history = {
    "train_loss": [0.6095, 0.5474, 0.5005, 0.4641, 0.4351, 0.2459],
    "eval_loss": [0.6164, 0.5593, 0.5160, 0.4824, 0.4557, 0.2808],
    "accuracy": [0.8889, 0.9000, 0.9056, 0.9167, 0.9167, 0.9222],
    "val_accuracy": [0.8167, 0.8333, 0.8333, 0.8333, 0.8500, 0.9167],
}

final_metrics = {
    "train_loss": history["train_loss"][-1],
    "validation_loss": history["eval_loss"][-1],
    "train_accuracy": history["accuracy"][-1],
    "validation_accuracy": history["val_accuracy"][-1],
}
```

Then ask TrainLens for a report and improvement ideas:

```python
%load_ext trainlens.magic.extension
%explain_training
%suggest_improvements
```

The CPU example above gives TrainLens enough evidence to report that both
training and validation loss decreased, final validation accuracy reached
`0.9167`, and no obvious validation drift appeared in the recorded run.

## Token Usage

Each magic call sends notebook evidence to the configured LLM provider. Token
usage depends on your notebook size, model, provider tokenizer, and generated
report length. The numbers below are estimates based on the CPU example, not a
billing record.

| Call | Estimated Input Tokens | Estimated Output Tokens | Estimated Total |
| --- | ---: | ---: | ---: |
| `%explain_training` | 1,000-2,000 | 2,500-3,500 | 3,500-5,500 |
| `%suggest_improvements` | 1,000-2,000 | 2,000-3,000 | 3,000-5,000 |
| Both calls | 2,000-4,000 | 4,500-6,500 | 6,500-10,500 |

## Python API

The notebook magics are wrappers around Python helpers:

```python
from trainlens import build_improvement_ideas, build_paper_report

paper = build_paper_report(globals())
ideas = build_improvement_ideas(globals())
```

## What TrainLens Sends

TrainLens summarizes visible notebook state, redacts likely secrets, sends the
summary to the configured LLM endpoint, and displays Markdown in the notebook.
The prompt tells the model to use only the supplied notebook evidence.

## License

Apache License 2.0. See [LICENSE](LICENSE).
