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

Compare two runs locally, without an LLM or tracking server:

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

In Jupyter, TrainLens can also inspect objects already in the notebook and
generate an LLM-backed report:

```python
%load_ext trainlens.magic.extension
%explain_training
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
