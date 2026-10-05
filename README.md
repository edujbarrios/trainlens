# TrainLens

**TrainLens turns training runs into evidence: what changed, what went wrong, and what experiment to run next.**

TrainLens is a lightweight, notebook-first toolkit for understanding ML training runs and
small experiment histories. It keeps train, validation, and held-out test evidence distinct,
compares runs with metric-aware semantics, and helps turn observations into controlled next
experiments. Core analysis is local and deterministic; LLM support is optional.

<p align="center">
  <a href="https://github.com/edujbarrios/trainlens/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/edujbarrios/trainlens/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://pypi.org/project/trainlens/0.14.3/"><img alt="PyPI 0.14.3" src="https://img.shields.io/badge/pypi-0.14.3-blue?logo=pypi"></a>
  <a href="pyproject.toml"><img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-blue"></a>
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-yellow"></a>
</p>

<a href="https://colab.research.google.com/github/edujbarrios/trainlens/blob/main/examples/quickstart.ipynb">
  <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
</a>

## Quickstart

```python
%pip install -q "trainlens[plots]"
```

TrainLens works with normal notebook state; no experiment-tracking server is required.

```python
history = {
    "train_loss": [1.31, 0.93, 0.66, 0.47, 0.34, 0.25],
    "val_loss":   [1.28, 0.91, 0.70, 0.58, 0.55, 0.57],
}

test_metrics = {"loss": 0.64, "accuracy": 0.75, "f1": 0.74}

from trainlens import analyze, plot_training_curves, render_report

analysis = analyze(globals())
plot_training_curves(globals())
print(render_report(analysis))
```

TrainLens preserves split semantics, detects evidence such as late validation degradation or
a validation-to-test gap, and understands common fine-tuning context including PEFT/LoRA,
trainable parameters, quantization, frozen components, and multimodal/VLM settings.

## Build an experiment history

Portable `TrainingRun` objects can live in a small local project instead of disappearing with
the notebook session.

```python
from trainlens import Project, select_checkpoint

project = Project(".trainlens")
entry = project.capture(
    globals(),
    name="lr-2e-5-seed-1",
    parameters={"learning_rate": 2e-5, "seed": 1},
)

selection = select_checkpoint(entry.run, metric="validation_loss")
print(selection.best_step, selection.suggested_patience)
```

As runs accumulate, TrainLens can rank them, identify Pareto-efficient configurations,
aggregate repeated seeds, compare observed improvements with between-run variability, and
learn parameter effects only from controlled pairs that differ in one recorded parameter.

## What TrainLens can do

- **Analyze train / validation / test evidence** without collapsing their roles.
- **Inspect fine-tuning configuration** across common Hugging Face, PEFT/LoRA, Keras,
  Lightning, PyTorch-style, and VLM workflows through dependency-light introspection.
- **Plot and monitor training** with learning curves, alerts, and framework callbacks.
- **Compare experiments** pairwise or through multi-run leaderboards with objectives,
  constraints, and Pareto fronts.
- **Reason about repeated runs** with mean/spread summaries and conservative signal-to-noise
  comparisons rather than overclaiming statistical significance.
- **Estimate parameter effects** from controlled one-parameter run pairs.
- **Select checkpoints deterministically** from training/validation trajectories while
  rejecting held-out test metrics for tuning.
- **Capture reproducibility evidence** such as Git revision, seed, environment, package
  versions, and dataset fingerprints.
- **Track resource and cost metrics** such as duration, throughput, memory, and estimated
  cost so they can participate in experiment objectives.
- **Store runs locally** with `Project`, or save/load portable run JSON directly.
- **Gate regressions in CI** with the `trainlens` command-line interface.
- **Extend analysis** with process-local analyzer plugins.
- **Use an optional LLM** for structured improvement plans whose evidence citations are
  checked against deterministic TrainLens evidence IDs.
- **Export reports** to Markdown, JSON, HTML, and optional PDF.

## Compare or gate runs from the CLI

```bash
trainlens compare baseline.json candidate.json
trainlens check candidate.json --against baseline.json --require 'f1>=0.85'
```

`trainlens check` exits non-zero on material regressions, failed metric gates, or missing
required metrics, so portable runs can be used directly in CI.

## Optional LLM layer

LLM support is deliberately downstream of local analysis. You can preview the exact sanitized
request, use any OpenAI-compatible endpoint, or request a structured plan with verified
evidence references:

```python
from trainlens import build_verified_improvement_plan

plan = build_verified_improvement_plan(globals(), provider=llm)
print(plan.recommendations)
print(plan.unsupported_evidence_ids)
```

Provider setup, prompt recipes, `prompt_text()`, `load_prompt()`, privacy controls, and request
previewing are documented in [LLM setup and privacy](docs/en/llm-and-privacy.md) and
[verified LLM plans](docs/en/verified-llm-plans.md).

## Documentation

- [Getting started](docs/en/getting-started.md)
- [Experiment workflows](docs/en/experiment-workflows.md)
- [Notebook workflows](docs/en/notebooks.md)
- [Python API](docs/en/python-api.md)
- [Monitoring](docs/en/monitoring.md)
- [Analyzer plugins](docs/en/analyzer-plugins.md)
- [LLM setup and privacy](docs/en/llm-and-privacy.md)
- [Exports and troubleshooting](docs/en/exports-and-troubleshooting.md)
- [Documentación en español](docs/es/README.md)

## Privacy and scope

Local analysis does not make an LLM request. When LLM support is enabled, TrainLens bounds
and redacts notebook evidence and keeps notebook-derived content separate from trusted
instructions. Generated explanations should still be verified against the underlying data.

TrainLens is an experiment-understanding layer, not a full MLOps platform, tracker, profiler,
or causal diagnosis system. It is designed for research and small-to-medium workflows where
the important training context already lives in Python and Jupyter.

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md) to contribute and [SECURITY.md](SECURITY.md) to report
vulnerabilities. TrainLens is licensed under the [Apache License 2.0](LICENSE).
