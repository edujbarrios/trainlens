# TrainLens

**Understand, compare, and document model-training runs from Jupyter.**

TrainLens reads the metrics and model objects already present in a notebook. It
can compare runs, detect common training problems, export reports, and use an
optional OpenAI-compatible LLM to explain the available evidence.

<p align="center">
  <a href="https://github.com/edujbarrios/trainlens/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/edujbarrios/trainlens/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://pypi.org/project/trainlens/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/trainlens?logo=pypi"></a>
  <a href="pyproject.toml"><img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-blue"></a>
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-yellow"></a>
</p>

## Install

```bash
pip install trainlens
```

## Small example

Suppose you trained a spam classifier on 2,000 short messages: 1,000 spam and
1,000 legitimate messages. Every run uses the same 80/20 split and random seed;
each experiment changes one design choice relative to the baseline. This single
Jupyter cell compares model quality and inference speed, then asks TrainLens to
explain the trade-offs:

```python
import os
from getpass import getpass

from trainlens import (
    PromptOptions,
    build_paper_report,
    compare_runs,
    render_run_comparison,
)

# 1. Select any OpenAI-compatible endpoint and the model used for the report.
# Remote provider example:
os.environ["TRAINLENS_LLM_BASE_URL"] = "https://api.openai.com/v1"
os.environ["TRAINLENS_LLM_MODEL"] = "your-model"
os.environ["TRAINLENS_LLM_API_KEY"] = getpass("LLM API key: ")

# Local model example with Ollama (use these values instead of the ones above):
# os.environ["TRAINLENS_LLM_BASE_URL"] = "http://localhost:11434/v1"
# os.environ["TRAINLENS_LLM_MODEL"] = "llama3.2"
# os.environ["TRAINLENS_LLM_API_KEY"] = "ollama"  # Required; stays local.
# LM Studio, vLLM, and llama.cpp also work when their OpenAI-compatible
# server is running.

# 2. Keep the dataset description and completed run evidence in the notebook.
dataset_note = (
    "Balanced spam dataset: 2,000 short messages, 1,000 spam and 1,000 "
    "legitimate; fixed 80/20 split and random seed across all experiments."
)
experiments = [
    (
        "experiment 1 | baseline",
        {
            "validation_loss": 0.52,
            "accuracy": 0.84,
            "f1": 0.82,
            "latency_ms": 8.1,
        },
    ),
    (
        "experiment 2 | lower learning rate",
        {
            "validation_loss": 0.47,
            "accuracy": 0.87,
            "f1": 0.86,
            "latency_ms": 8.1,
        },
    ),
    (
        "experiment 3 | add dropout",
        {
            "validation_loss": 0.45,
            "accuracy": 0.88,
            "f1": 0.89,
            "latency_ms": 8.2,
        },
    ),
    (
        "experiment 4 | smaller hidden layer",
        {
            "validation_loss": 0.58,
            "accuracy": 0.82,
            "f1": 0.80,
            "latency_ms": 5.6,
        },
    ),
]

# These named series become part of the TrainLens notebook context.
experiment_validation_loss = [metrics["validation_loss"] for _, metrics in experiments]
experiment_accuracy = [metrics["accuracy"] for _, metrics in experiments]
experiment_f1 = [metrics["f1"] for _, metrics in experiments]
experiment_latency_ms = [metrics["latency_ms"] for _, metrics in experiments]

# 3. Compare every run with the baseline using deterministic TrainLens analysis.
baseline_name, baseline_metrics = experiments[0]
for experiment_name, experiment_metrics in experiments[1:]:
    comparison = compare_runs(
        baseline_metrics,
        experiment_metrics,
        baseline_name=baseline_name,
        experiment_name=experiment_name,
    )
    print(render_run_comparison(comparison))

# 4. Ask the selected LLM for a concise, evidence-first TrainLens diagnosis.
prompt_options = PromptOptions(
    prompt_name="training_diagnosis",
    objective=(
        "Compare quality and latency across all four experiments, identify the "
        "best quality run and fastest run, and propose one controlled next experiment."
    ),
    tone="short, clear, and evidence-first",
)
report = build_paper_report(globals(), prompt_options=prompt_options)
print(report.markdown)
```

TrainLens recognizes that lower loss and latency are improvements, while higher
accuracy and F1 are improvements. The results make the trade-off visible:
experiment 3 has the best model quality, but experiment 4 is faster at the cost
of worse predictive metrics. The final call sends the redacted notebook context
to the configured model for a short diagnosis.

The LLM workflow requires an OpenAI-compatible HTTP endpoint, but it does not
have to be an external service. You can use a remote provider or a locally
running model through Ollama, LM Studio, vLLM, or llama.cpp. Local comparison,
monitoring, experiment planning, and export remain deterministic and make no
LLM request.

> [!CAUTION]
> **Human oversight required:** TrainLens is intended to support understanding
> training results and making better-informed decisions—not to replace a human
> reviewer. LLM-generated explanations can contain errors, omissions, or biases
> inherited from a model's training data and design. Treat every recommendation
> as assistance for the programmer, verify it against the underlying evidence,
> and do not use it as the sole basis for consequential decisions.

## Documentation

The complete guide covers notebook setup, framework adapters, the Python API,
monitoring, prompts, privacy, exports, and troubleshooting:

- [English documentation](docs/en/README.md)
- [Documentación en español](docs/es/README.md)
- [Documentation index](docs/README.md)

A dedicated documentation website is planned. Until it is published, the
versioned Markdown files in `docs/` are the canonical guide.

## Scope

TrainLens is a lightweight notebook reporting layer, not a full MLOps platform.
It works best for small research workflows where experiment context lives in
Python variables and conclusions should remain easy to review.

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md) to contribute and
[SECURITY.md](SECURITY.md) to report vulnerabilities.

TrainLens is licensed under the [Apache License 2.0](LICENSE).
