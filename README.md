# TrainLens

**See what changed between training runs without leaving your notebook.**

TrainLens is a lightweight Python library for comparing training runs, spotting common
training problems, planning the next experiment, exporting results, and optionally asking
an OpenAI-compatible model to explain the evidence.

<p align="center">
  <a href="https://github.com/edujbarrios/trainlens/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/edujbarrios/trainlens/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://pypi.org/project/trainlens/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/trainlens?logo=pypi"></a>
  <a href="pyproject.toml"><img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-blue"></a>
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-yellow"></a>
</p>

## Try it in a notebook in 20 seconds

<a href="https://colab.research.google.com/github/edujbarrios/trainlens/blob/main/examples/quickstart.ipynb">
  <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
</a>

First cell:

```python
%pip install -q trainlens
```

Second cell:

```python
from trainlens import compare_runs

compare_runs(
    {"loss": 0.52, "accuracy": 0.84},
    {"loss": 0.41, "accuracy": 0.89},
    baseline_name="baseline",
    experiment_name="new run",
)
```

That is enough. In Jupyter, TrainLens renders the result as Markdown automatically:

> **Summary**
> - Material improvement detected in accuracy, loss.
>
> | Metric | Baseline | Experiment | Delta | Direction |
> | --- | ---: | ---: | ---: | --- |
> | accuracy | 0.84 | 0.89 | +0.05 | improved |
> | loss | 0.52 | 0.41 | -0.11 | improved |

No model provider, API key, callback, or experiment tracker is required for local comparison.

## A few useful snippets

Keep the structured result when you want to inspect it in code:

```python
comparison = compare_runs(old_metrics, new_metrics)
comparison.improvements
comparison.regressions
```

Render the same comparison anywhere Markdown is useful:

```python
markdown = comparison.to_markdown()
print(markdown)
```

Export a comparison to another format:

```python
from trainlens import render_report

html = render_report(comparison, format="html")
json_text = render_report(comparison, format="json")
```

TrainLens also understands richer `TrainingRun` objects, including metric trajectories and
configuration changes, when you need more than two dictionaries.

## Ask an LLM for an improvement plan

The LLM is optional. TrainLens first performs deterministic local analysis, then sends a
bounded, redacted evidence summary to the model. You can use a remote provider or a local
OpenAI-compatible server.

### 1. Load the model provider

```python
import os
from trainlens import OpenAICompatibleProvider

llm = OpenAICompatibleProvider.from_values(
    base_url="https://your-provider.example/v1",
    model="your-model",
    api_key=os.environ["YOUR_LLM_API_KEY"],
)
```

For a local server such as Ollama, LM Studio, vLLM, or llama.cpp, the same API works and an
API key is usually unnecessary:

```python
llm = OpenAICompatibleProvider.from_values(
    base_url="http://localhost:11434/v1",
    model="your-local-model",
)
```

If you prefer environment variables, configure `TRAINLENS_LLM_BASE_URL`,
`TRAINLENS_LLM_MODEL`, and optionally `TRAINLENS_LLM_API_KEY`, then use:

```python
llm = OpenAICompatibleProvider.from_env()
```

### 2. Define what you want the model to analyze

```python
from trainlens import improvement_plan_prompt

prompt = improvement_plan_prompt(
    objective=(
        "Find the three highest-impact improvements for the next run. "
        "Prefer low-cost, controlled changes and explain the evidence for each one."
    ),
    model_family="vision-language model",
    focus_areas=(
        "generalization gap",
        "learning-rate schedule",
        "adapter capacity",
    ),
)
```

The default improvement recipe asks for evidence, the proposed change, rationale, expected
effect, cost/risk, confidence, a measurable success criterion, and one concrete next run.
It also tells the model not to invent missing hyperparameters.

### 3. Inspect the exact prompt before sending anything

```python
from trainlens import preview_llm_request

preview = preview_llm_request(
    globals(),
    mode="improvement_ideas",
    provider=llm,
    prompt_options=prompt,
)

print(preview.system_prompt)  # trusted analysis instructions
print(preview.user_prompt)    # sanitized notebook evidence
```

`preview_llm_request()` performs no network request. The preview is built by the same code
path used for the real OpenAI-compatible request, so what you inspect is what TrainLens will
send, excluding the API key.

### 4. Generate the improvement report

```python
from trainlens import build_improvement_ideas

report = build_improvement_ideas(
    globals(),
    provider=llm,
    prompt_options=prompt,
)

report  # renders as Markdown in Jupyter
```

TrainLens detects the metrics, model/trainer evidence, local signals, and deterministic
recommendations already present in the notebook. The LLM explains and prioritizes that
evidence rather than replacing the local analysis with an ungrounded interpretation.

## What else can TrainLens do?

- **Compare runs** with metric-aware improvement/regression semantics.
- **Inspect notebook training context** without wiring a full MLOps stack.
- **Monitor training** with built-in and extensible alert detectors.
- **Plan the next experiment** with constraints, objectives, and Pareto-aware helpers.
- **Save and load portable runs** for repeatable comparisons.
- **Export reports** to Markdown, JSON, HTML, and optional PDF.
- **Use an optional LLM** to explain deterministic TrainLens findings through any
  OpenAI-compatible endpoint, including local servers.

The local analysis features are deterministic and do not make an LLM request.

## Want the deeper API?

The README intentionally stays small. The versioned docs contain the advanced workflows:

- [Getting started](docs/en/getting-started.md)
- [Notebook workflows](docs/en/notebooks.md)
- [Python API](docs/en/python-api.md)
- [Monitoring](docs/en/monitoring.md)
- [LLM setup and privacy](docs/en/llm-and-privacy.md)
- [Exports and troubleshooting](docs/en/exports-and-troubleshooting.md)
- [Documentación en español](docs/es/README.md)

## Privacy

LLM support is optional. When enabled, TrainLens minimizes and bounds notebook evidence,
redacts secrets, and keeps notebook-derived evidence separate from trusted system
instructions. Treat generated explanations as assistance and verify them against the
underlying training evidence.

## Scope

TrainLens is a lightweight notebook reporting and experiment-understanding layer, not a
full MLOps platform. It is designed for research and small-to-medium workflows where the
important training context already lives in Python and Jupyter.

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md) to contribute and [SECURITY.md](SECURITY.md) to
report vulnerabilities.

TrainLens is licensed under the [Apache License 2.0](LICENSE).
