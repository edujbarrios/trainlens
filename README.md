# TrainLens

**TrainLens turns training runs into evidence: what changed, what went wrong, and what experiment to run next.**

TrainLens is a lightweight, notebook-first toolkit for understanding ML training runs and small
experiment histories. It keeps train, validation, and held-out test evidence distinct, compares
runs with metric-aware semantics, explains aggregate dataset context, and helps turn observations
into controlled next experiments. Core analysis is local and deterministic; LLM support is
optional.

**New in 0.16.0: TrainLens can be used as the evidence layer inside AutoResearch-style skills and
coding agents.** Claude, Codex, Cursor, or another already-running agent can consume deterministic
TrainLens context, reason with its own model, and return a structured experiment plan. In Agent
Mode, TrainLens makes **no LLM call**, needs **no provider SDK**, and requires **no API key**.

<p align="center">
  <a href="https://github.com/edujbarrios/trainlens/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/edujbarrios/trainlens/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://pypi.org/project/trainlens/0.16.0/"><img alt="PyPI 0.16.0" src="https://img.shields.io/badge/pypi-0.16.0-blue?logo=pypi"></a>
  <a href="pyproject.toml"><img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-blue"></a>
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-yellow"></a>
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/edujbarrios/trainlens/blob/main/examples/quickstart.ipynb">
    <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
  </a>
</p>

## Install

```bash
pip install trainlens
```

For plots:

```bash
pip install "trainlens[plots]"
```

No tracking server or hosted TrainLens service is required.

## Analyze a training run locally

Start from the objects you already have after training:

```python
history = {
    "train_loss":     [1.31, 0.93, 0.66, 0.47, 0.34, 0.25],
    "val_loss":       [1.28, 0.91, 0.70, 0.58, 0.55, 0.57],
    "train_accuracy": [0.55, 0.68, 0.79, 0.87, 0.92, 0.95],
    "val_accuracy":   [0.54, 0.66, 0.75, 0.80, 0.82, 0.81],
}

test_metrics = {
    "loss": 0.64,
    "accuracy": 0.75,
    "f1": 0.74,
    "precision": 0.77,
    "recall": 0.72,
}
```

Analyze the notebook state and plot learning curves:

```python
from trainlens import analyze, plot_training_curves, render_report

analysis = analyze(globals())
plot_training_curves(globals())
print(render_report(analysis))
```

TrainLens can preserve final test metrics, identify widening train/validation gaps, notice late
validation degradation, inspect common fine-tuning configuration, and keep held-out test evidence
out of tuning decisions.

## Explain the dataset behind the metrics

`explain_dataset()` creates a deterministic aggregate profile without requiring pandas, Hugging
Face Datasets, Pillow, NumPy, PyTorch, or an external service:

```python
from trainlens import explain_dataset

dataset_context = explain_dataset(
    {
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset,
    },
    target="label",
)
```

TrainLens understands tabular, text, image, and image-centered multimodal datasets. Visual context
is metadata-only: pixels, image bytes, private paths, and filenames are not rendered into the
dataset explanation.

## Agent Mode: use TrainLens inside AutoResearch skills

Agent Mode separates **evidence** from **reasoning**:

```text
training state / portable run
            │
            ▼
        TrainLens
  deterministic analysis
            │
            ▼
       AgentContext
 metrics · signals · evidence IDs
 training profile · dataset context
 instructions · output schema
            │
            ▼
 Claude / Codex / Cursor / other agent
       agent-owned reasoning
            │
            ▼
      structured plan
            │
            ▼
        TrainLens
   evidence-ID verification
```

The agent is already the model runtime. TrainLens does not call another model behind it.

```python
from trainlens import build_agent_context, verify_agent_plan

context = build_agent_context(
    globals(),
    dataset_explanation=dataset_context,
)

# Give this bounded context to the coding/research agent that is already running.
print(context.to_markdown())

# The external agent returns JSON matching context.output_schema.
plan = verify_agent_plan(agent_response, context=context)
print(plan.is_fully_supported)
print(plan.unsupported_evidence_ids)
```

`verify_agent_plan()` verifies that recommendations cite real deterministic TrainLens evidence.
It does not claim that a recommendation is scientifically correct; the agent and human reviewer
still own experimental judgment.

### Agent Mode from the terminal

Skills and coding agents often work from files rather than live notebook memory. TrainLens can
build context from a portable `TrainingRun` JSON:

```bash
trainlens agent-context candidate.json --format json > agent-context.json
```

The agent reads `agent-context.json`, reasons using its own model, and writes `plan.json`. Then:

```bash
trainlens verify-agent-plan plan.json --context agent-context.json
```

This workflow requires no `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or TrainLens-specific LLM
configuration. See [Agent Mode](docs/en/agent-mode.md) and the
[example Agent Skill](examples/agent-skill/SKILL.md).

## Turn diagnosis into controlled experiments

TrainLens can preserve runs, select checkpoints from validation evidence, compare experiments,
and build a small local experiment history:

```python
from trainlens import Project, select_checkpoint

project = Project(".trainlens")
entry = project.capture(
    globals(),
    name="lr-2e-5-seed-1",
    parameters={
        "learning_rate": 2e-5,
        "seed": 1,
        "lora_r": 16,
    },
)

selection = select_checkpoint(entry.run, metric="validation_loss")
print(selection.best_step)
print(selection.best_value)
print(selection.suggested_patience)
```

As runs accumulate, TrainLens can rank them with objectives and constraints, mark Pareto-efficient
configurations, aggregate repeated seeds, compare observed improvements with between-run
variability, and estimate parameter effects from controlled pairs.

## Optional: let TrainLens call an LLM directly

Agent Mode does not replace the existing provider integration. If you want TrainLens itself to
request a generated explanation, you can still inject an OpenAI-compatible provider:

```python
from getpass import getpass
from trainlens import OpenAICompatibleProvider

llm = OpenAICompatibleProvider.from_values(
    base_url="https://your-provider.example/v1",
    model="your-model",
    api_key=getpass("LLM API key: "),
)
```

Preview the exact sanitized request before sending it:

```python
from trainlens import preview_llm_request

preview = preview_llm_request(
    globals(),
    mode="improvement_ideas",
    provider=llm,
    dataset_explanation=dataset_context,
)

print(preview.system_prompt)
print(preview.user_prompt)
```

For machine-checkable provider workflows, `build_verified_improvement_plan()` continues to use
the same stable deterministic evidence IDs as Agent Mode.

## What TrainLens can do

- **Analyze train / validation / test evidence** without collapsing their roles.
- **Prepare provider-free AgentContext** for AutoResearch-style skills and coding agents.
- **Verify structured agent plans** against stable deterministic evidence IDs.
- **Explain tabular, image, and multimodal datasets locally** with aggregate privacy-preserving
  context.
- **Inspect fine-tuning configuration** across common Hugging Face, PEFT/LoRA, Keras, Lightning,
  PyTorch-style, and VLM workflows through dependency-light introspection.
- **Plot and monitor training** with learning curves, alerts, and framework callbacks.
- **Select checkpoints deterministically** from training/validation trajectories.
- **Compare experiments** pairwise or through multi-run leaderboards, objectives, constraints,
  and Pareto fronts.
- **Reason about repeated runs** with mean/spread summaries and conservative comparisons.
- **Estimate parameter effects** from controlled one-parameter run pairs.
- **Capture reproducibility evidence** such as Git revision, seed, environment, package versions,
  and dataset fingerprints.
- **Track resource and cost metrics** such as duration, throughput, memory, and estimated cost.
- **Store runs locally** with `Project`, or save/load portable run JSON directly.
- **Gate regressions in CI** with the `trainlens` command-line interface.
- **Extend analysis** with process-local analyzer plugins.
- **Use optional LLMs** with request previewing, privacy controls, prompt recipes, and
  evidence-verified structured plans.
- **Export reports** to Markdown, JSON, HTML, and optional PDF.

## CLI

```bash
trainlens compare baseline.json candidate.json
trainlens check candidate.json --against baseline.json --require 'f1>=0.85'
trainlens agent-context candidate.json --format json
trainlens verify-agent-plan plan.json --context agent-context.json
```

## Documentation

- [Getting started](docs/en/getting-started.md)
- [Agent Mode and AutoResearch skills](docs/en/agent-mode.md)
- [Dataset-aware analysis](docs/en/dataset-context.md)
- [Experiment workflows](docs/en/experiment-workflows.md)
- [Notebook workflows](docs/en/notebooks.md)
- [Python API](docs/en/python-api.md)
- [Monitoring](docs/en/monitoring.md)
- [Analyzer plugins](docs/en/analyzer-plugins.md)
- [LLM setup and privacy](docs/en/llm-and-privacy.md)
- [Verified LLM plans](docs/en/verified-llm-plans.md)
- [Exports and troubleshooting](docs/en/exports-and-troubleshooting.md)
- [Documentación en español](docs/es/README.md)

## Privacy and scope

Local model and dataset analysis does not make an LLM request. Agent Mode also makes no LLM
request: it only prepares bounded evidence for an already-running external agent and verifies the
returned evidence references. Dataset explanations are aggregate profiles and do not contain raw
feature rows, pixels, image bytes, or private image paths.

When direct LLM support is enabled, TrainLens bounds and redacts notebook evidence, and dataset
context is only included when a `DatasetExplanation` is explicitly supplied.

TrainLens is an experiment-understanding and evidence layer, not a full MLOps platform, tracker,
profiler, AutoML engine, or causal diagnosis system. Generated or agent-proposed conclusions
should always be checked against the underlying data and experimental design.

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md) to contribute and [SECURITY.md](SECURITY.md) to report
vulnerabilities. TrainLens is licensed under the [Apache License 2.0](LICENSE).
