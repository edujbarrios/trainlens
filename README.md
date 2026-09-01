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

Suppose you are training a spam classifier on 2,000 short messages: 1,000 spam
and 1,000 legitimate messages. The same 80/20 train-validation split and random
seed are used in every experiment, so only the learning rate changes.

```python
from trainlens import compare_runs, render_run_comparison

experiments = {
    "experiment 1 · baseline · lr=1e-3": {
        "validation_loss": 0.52,
        "accuracy": 0.84,
    },
    "experiment 2 · lr=5e-4": {
        "validation_loss": 0.48,
        "accuracy": 0.87,
    },
    "experiment 3 · lr=2e-4": {
        "validation_loss": 0.44,
        "accuracy": 0.90,
    },
    "experiment 4 · lr=1e-4": {
        "validation_loss": 0.46,
        "accuracy": 0.89,
    },
}

baseline_name, baseline_metrics = next(iter(experiments.items()))
for experiment_name, experiment_metrics in list(experiments.items())[1:]:
    comparison = compare_runs(
        baseline_metrics,
        experiment_metrics,
        baseline_name=baseline_name,
        experiment_name=experiment_name,
    )
    print(render_run_comparison(comparison))
```

TrainLens reports the metric deltas and recognizes that lower loss and higher
accuracy are improvements. Here, experiment 3 is the strongest result; reducing
the learning rate again in experiment 4 does not improve it further.

In Jupyter, you can ask TrainLens for a short evidence-based explanation. Keep
the dataset note and metric series in the inspected notebook context:

```python
from trainlens import PromptOptions, build_paper_report

dataset_note = (
    "Balanced spam dataset: 2,000 short messages, 1,000 spam and 1,000 "
    "legitimate; fixed 80/20 split and random seed across all experiments."
)
experiment_validation_loss = [0.52, 0.48, 0.44, 0.46]
experiment_accuracy = [0.84, 0.87, 0.90, 0.89]

prompt = PromptOptions(
    prompt_name="training_diagnosis",
    objective=(
        "Explain which experiment performed best, describe the trend across "
        "the four runs, and suggest one controlled next experiment."
    ),
    tone="short, clear, and evidence-first",
)

report = build_paper_report(globals(), prompt_options=prompt)
print(report.markdown)
```

The LLM workflow requires an OpenAI-compatible endpoint. Local comparison,
monitoring, experiment planning, and export remain deterministic and do not
contact an external service.

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
