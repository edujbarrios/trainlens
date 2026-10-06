# TrainLens

**TrainLens turns ML training runs into evidence: what changed, what went wrong, and what experiment to run next.**

TrainLens is lightweight, notebook-first, and local by default. Its core analysis is deterministic; LLM reasoning is optional.

<p align="center">
  <a href="https://github.com/edujbarrios/trainlens/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/edujbarrios/trainlens/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://pypi.org/project/trainlens/0.17.0/"><img alt="PyPI 0.17.0" src="https://img.shields.io/badge/pypi-0.17.0-blue?logo=pypi"></a>
  <a href="pyproject.toml"><img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-blue"></a>
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-yellow"></a>
</p>

## Start with the notebook

The fastest way to understand TrainLens is the end-to-end example:

[**Open the TrainLens quickstart notebook in Google Colab**](https://colab.research.google.com/github/edujbarrios/trainlens/blob/main/examples/quickstart.ipynb)

It shows dataset context, training diagnosis, plots, experiment evidence, LLM request previewing, and evidence-backed next-step planning.

## Install

```bash
pip install trainlens
```

For plots:

```bash
pip install "trainlens[plots]"
```

No tracking server or hosted TrainLens service is required.

## Choose how you want to use TrainLens

| Workflow | Who does the reasoning? | Extra LLM setup inside TrainLens? |
| --- | --- | --- |
| **Local analysis** | TrainLens deterministic analyzers | No |
| **Agent Mode** | The surrounding agent: Claude, Codex, Cursor, etc. | Usually no |
| **LLM reports** | An LLM provider called directly by TrainLens | Yes, optional provider configuration |

### 1. Local deterministic analysis

```python
from trainlens import analyze, plot_training_curves, render_report

analysis = analyze(globals())
plot_training_curves(globals())
print(render_report(analysis))
```

TrainLens keeps training, validation, and held-out test evidence distinct and turns notebook state into structured observations and recommendations.

### 2. Agent Mode: TrainLens as the evidence layer

**TrainLens can be used inside AutoResearch-style skills and coding agents.**

```python
from trainlens import build_agent_context, verify_agent_plan

context = build_agent_context(globals())
print(context.to_markdown())

plan = verify_agent_plan(agent_response, context=context)
print(plan.is_fully_supported)
```

In Agent Mode, TrainLens prepares bounded deterministic context, stable evidence IDs, instructions, and an output schema. The surrounding agent performs the reasoning and tool execution.

Agent Mode can therefore use **whatever LLM and provider are available to the host agent**. In the common case, Claude, Codex, Cursor, or another coding/research agent already owns that model connection, so there is no reason to configure a second LLM provider inside TrainLens.

```text
training state -> TrainLens evidence -> Claude / Codex / Cursor -> structured plan -> TrainLens verification
```

Terminal-first agents can use portable run JSON:

```bash
trainlens agent-context candidate.json --format json > agent-context.json
trainlens verify-agent-plan plan.json --context agent-context.json
```

See [Agent Mode](docs/en/agent-mode.md) and the reusable [Agent Skill example](examples/agent-skill/SKILL.md).

### 3. No agent? TrainLens can generate LLM reports directly

Agent Mode is optional. **TrainLens can also call an LLM provider itself** to generate training reports, diagnoses, or improvement ideas from its sanitized deterministic evidence.

```python
from getpass import getpass
from trainlens import OpenAICompatibleProvider, build_llm_report

llm = OpenAICompatibleProvider.from_values(
    base_url="https://your-provider.example/v1",
    model="your-model",
    api_key=getpass("LLM API key: "),
)

report = build_llm_report(globals(), provider=llm)
print(report.markdown)
```

You can inspect the exact sanitized request before sending it with `preview_llm_request()`. TrainLens also supports evidence-verified structured improvement plans through `build_verified_improvement_plan()`.

See [LLM setup and privacy](docs/en/llm-and-privacy.md) and [Verified LLM plans](docs/en/verified-llm-plans.md).

## Dataset context

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

TrainLens can summarize tabular, text, image, and image-centered multimodal datasets without sending raw rows, pixels, image bytes, private paths, or filenames to an LLM.

## Experiments, runs, and CI

TrainLens can save portable runs, compare experiments, select validation-based checkpoints, aggregate repeated seeds, estimate controlled parameter effects, build Pareto-aware leaderboards, and gate regressions in CI.

```bash
trainlens compare baseline.json candidate.json
trainlens check candidate.json --against baseline.json --require 'f1>=0.85'
```

## Core idea

TrainLens is the **evidence layer**, not AutoML:

- deterministic analysis stays local by default;
- Agent Mode lets an existing agent reason over TrainLens evidence using the agent's own LLM/provider;
- direct LLM reports remain available when no agent is involved;
- held-out test evidence is kept out of tuning decisions;
- generated recommendations should still be checked against the underlying data and experimental design.

## Documentation

- [Quickstart notebook](examples/quickstart.ipynb)
- [Getting started](docs/en/getting-started.md)
- [Agent Mode and AutoResearch skills](docs/en/agent-mode.md)
- [Dataset-aware analysis](docs/en/dataset-context.md)
- [Experiment workflows](docs/en/experiment-workflows.md)
- [LLM setup and privacy](docs/en/llm-and-privacy.md)
- [Python API](docs/en/python-api.md)
- [Documentation in Spanish](docs/es/README.md)

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md) to contribute and [SECURITY.md](SECURITY.md) to report vulnerabilities. TrainLens is licensed under the [Apache License 2.0](LICENSE).
