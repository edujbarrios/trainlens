# TrainLens

**Understand what happened during ML training and what to try next.**

TrainLens analyzes metrics and training state from Python/Jupyter notebooks or saved runs. It produces local, deterministic evidence and reports; it does **not** train models or require a tracking server. LLM-generated explanations are optional.

<p align="center">
  <a href="https://github.com/edujbarrios/trainlens/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/edujbarrios/trainlens/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://pypi.org/project/trainlens/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/trainlens?logo=pypi"></a>
  <a href="pyproject.toml"><img alt="Python 3.11–3.14" src="https://img.shields.io/badge/python-3.11--3.14-blue"></a>
  <a href="LICENSE"><img alt="Apache-2.0 license" src="https://img.shields.io/badge/license-Apache--2.0-yellow"></a>
</p>

## Quick start

Install with Python 3.11–3.14:

```bash
python -m pip install trainlens
```

Analyze a training history without an API key:

```python
from trainlens import analyze, plot_training_curves, render_report

namespace = {
    "history": {
        "train_loss": [1.0, 0.7, 0.4],
        "val_loss": [1.1, 0.75, 0.52],
    }
}

result = analyze(namespace)
print(render_report(result))
plot_training_curves(namespace, metrics=("loss",))
```

In a real notebook, pass `globals()` to inspect available training metrics, models, and trainers. TrainLens uses the evidence it finds; it does not run training or fetch remote experiment data.

**Prefer a guided example?** [Run the quickstart notebook in Google Colab](https://colab.research.google.com/github/edujbarrios/trainlens/blob/main/examples/quickstart.ipynb).

## How it works

1. **Collect evidence:** inspect metric histories and compatible objects already in the notebook, or read a portable run JSON.
2. **Analyze locally:** normalize metrics, distinguish training/validation/test, and generate deterministic findings.
3. **Use the results:** read a report, plot curves, compare runs, or optionally send bounded evidence to an agent or LLM.

There are three ways to use TrainLens:

| Mode | Who interprets the evidence? | LLM setup in TrainLens |
| --- | --- | --- |
| **Local analysis** | TrainLens deterministic analyzers | None |
| **Agent Mode** | Your existing coding/research agent | None |
| **Direct LLM report** | Your configured OpenAI-compatible provider | Required |

### Agent Mode

TrainLens prepares evidence with IDs for an external agent (such as Codex, Claude, or Cursor). The agent proposes a plan; TrainLens can check whether its cited evidence IDs exist. **This is not a guarantee that the plan is scientifically correct.**

```python
from trainlens import build_agent_context

context = build_agent_context(namespace)
print(context.to_markdown())  # Give this to your existing agent
```

See [Agent Mode and plan verification](docs/en/agent-mode.md).

### Direct LLM reports (optional)

Configure `TRAINLENS_LLM_BASE_URL` and `TRAINLENS_LLM_MODEL`, plus `TRAINLENS_LLM_API_KEY` if the endpoint requires it.

```python
from trainlens import build_llm_report

report = build_llm_report(namespace)
print(report.markdown)
```

TrainLens calls the configured provider in this mode. See [LLM setup, request previews, and privacy](docs/en/llm-and-privacy.md).

## Compare saved runs

Portable `TrainingRun` JSON files let you compare experiments or check regressions without a server:

```bash
trainlens compare baseline.json candidate.json
trainlens check candidate.json --against baseline.json --require 'f1>=0.85'
```

See [saving runs and experiment workflows](docs/en/experiment-workflows.md) for how to create these files.

## Scope and privacy

- Local analysis, plotting, and Agent Mode do not make LLM requests from TrainLens. An **external agent** may have its own network and data policies.
- Direct LLM reports send a sanitized, bounded context to your chosen endpoint. Redaction is best-effort: review the request before sending sensitive data.
- Plots (Matplotlib) and PDF export (ReportLab) are included in the standard install. Training frameworks such as PyTorch, TensorFlow, and Transformers are **not** installed by TrainLens.
- TrainLens supports evidence-based inspection, not AutoML, causal proof, or a replacement for a full experiment tracker. Do not use held-out test results to choose training changes.

## Documentation

- [Getting started and notebook commands](docs/en/getting-started.md)
- [Python API](docs/en/python-api.md)
- [Agent Mode](docs/en/agent-mode.md)
- [Datasets and multimodal context](docs/en/dataset-context.md)
- [Experiments, exports, and troubleshooting](docs/en/experiment-workflows.md)
- [Documentation in Spanish / Documentación en español](docs/es/README.md)

For contributions, see [CONTRIBUTING.md](CONTRIBUTING.md). Report vulnerabilities via [SECURITY.md](SECURITY.md). Licensed under [Apache-2.0](LICENSE).
