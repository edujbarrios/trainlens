# TrainLens

TrainLens turns notebook training state into LLM-written reports inside
Jupyter. It reads variables already in memory, such as `history`,
`training_params`, `model_name`, dataset notes, metrics, and trace logs, then
sends a compact redacted context to an OpenAI-compatible chat endpoint.

<p align="center">
  <a href="https://github.com/edujbarrios/trainlens/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/edujbarrios/trainlens/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://pypi.org/project/trainlens/"><img alt="PyPI" src="https://img.shields.io/pypi/v/trainlens?label=PyPI&logo=pypi"></a>
  <a href="pyproject.toml"><img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-blue"></a>
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-yellow"></a>
</p>

<p align="center">
  <strong>Actively maintained by Eduardo J. Barrios.</strong><br>
  For issues or improvements, open a pull request or email
  <a href="mailto:edujbarrios@outlook.com">edujbarrios@outlook.com</a>.
</p>

## Install

```bash
pip install trainlens
```

## Configure

TrainLens works with OpenAI-compatible APIs:

```python
import os

os.environ["TRAINLENS_LLM_BASE_URL"] = "https://api.openai.com/v1"
os.environ["TRAINLENS_LLM_API_KEY"] = "your-api-key"
os.environ["TRAINLENS_LLM_MODEL"] = "gpt-5.4-mini"
```

Local models work too. For Ollama:

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

The same pattern works with LM Studio, vLLM, llama.cpp server, and other
OpenAI-compatible local servers.

## Notebook Quickstart

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

Then run:

```python
%load_ext trainlens.magic.extension

# Scientific paper-style training report
%explain_training

# Evidence-backed experiment plan
%suggest_improvements
```

| Magic | Output |
| --- | --- |
| `%explain_training` | Scientific paper-style report with results, discussion, limitations, and LLM provenance. |
| `%suggest_improvements` | Follow-up experiment plan grounded in the same notebook evidence. |

## Python API

```python
from trainlens import build_improvement_ideas, build_paper_report

paper = build_paper_report()
ideas = build_improvement_ideas()
```

Inside Jupyter, helpers read the active notebook namespace automatically.
Outside IPython, pass an explicit dictionary-like namespace.

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

Formula: `(input_tokens * input_price_per_1M + output_tokens * output_price_per_1M) / 1_000_000`.

## Privacy

TrainLens summarizes visible notebook state, redacts likely secrets, truncates
large literals, and sends only the resulting context to the configured endpoint.
Do not store real API keys in notebooks or committed files.

## License

Apache License 2.0. See [LICENSE](LICENSE).
