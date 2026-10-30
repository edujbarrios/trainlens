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

Analyze a training history locally, without an API key:

```python
from trainlens import analyze, plot_training_curves, render_report

# Option 1: use metrics already available in Python.
history = {
    "train_loss": [1.0, 0.7, 0.4],
    "val_loss": [1.1, 0.75, 0.52],
}

# Option 2: replace the history above with metrics from a plain JSON file.
# import json
# from pathlib import Path
# history = json.loads(Path("history.json").read_text(encoding="utf-8"))

# Give TrainLens a dictionary of variables to inspect.
training_state = {"history": history}
analysis = analyze(training_state)
print(render_report(analysis))  # Local findings and recommendations
plot_training_curves(training_state, metrics=("loss",))
```

The `"history"` key tells TrainLens where to find the metrics. A plain `history.json` should contain the same metric dictionary shown above. In a Jupyter notebook, you can simply use `analyze(globals())` to inspect the variables already defined there.

Portable **TrainLens run JSON** (created with `save_run()`) has a different structure; use `load_run()` with the run-comparison or Agent Mode APIs rather than loading it as a plain history dictionary.

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

# Reuse the training_state from Quick start, or pass globals() in a notebook.
context = build_agent_context(training_state)
print(context.to_markdown())  # Share this evidence with your agent

# For a portable TrainLens run JSON created with save_run():
# from trainlens import build_agent_context_from_run, load_run
# context = build_agent_context_from_run(load_run("candidate.json"))
```

See [Agent Mode and plan verification](docs/en/agent-mode.md).

### Direct LLM reports (optional): Llama, prompts, and API keys

TrainLens does **not** download, train, or run Llama models itself. It sends an
OpenAI-compatible request to `<TRAINLENS_LLM_BASE_URL>/chat/completions`.
You can use a local Llama server (for example, Ollama) or a hosted provider with
a compatible API. This step is **optional**: local analysis and Agent Mode need
no TrainLens API key.

#### 1. Run Llama locally with Ollama (no API key)

Install and start [Ollama](https://ollama.com/download), then download a model:

```bash
ollama pull llama3.2
ollama list  # Check the exact installed model name
```

If Ollama is not already running as a service, start `ollama serve` in a
separate terminal. Set these variables **before starting** Python/Jupyter
(Bash/Zsh):

```bash
export TRAINLENS_LLM_BASE_URL="http://localhost:11434/v1"
export TRAINLENS_LLM_MODEL="llama3.2"
unset TRAINLENS_LLM_API_KEY  # Local Ollama normally needs no credentials
```

`/v1` is the OpenAI-compatible base URL; **do not** append
`/chat/completions` yourself. The model value must match the model exposed
by your server. LM Studio, vLLM, and llama.cpp also work **if** configured to
serve a compatible chat-completions API; use their own URL and model name.

#### 2. Configure a hosted model and its API key

Get the **base URL**, **model ID**, and **API key** from your chosen provider.
TrainLens reads the following environment variables (it does not read
`OPENAI_API_KEY` automatically):

| Variable | Required? | Purpose |
| --- | --- | --- |
| `TRAINLENS_LLM_BASE_URL` | Yes | Provider's OpenAI-compatible base URL, usually ending in `/v1` |
| `TRAINLENS_LLM_MODEL` | Yes | Exact model ID supported by that endpoint (including hosted Llama models) |
| `TRAINLENS_LLM_API_KEY` | Only when required | Sent as the `Authorization: Bearer …` header |
| `TRAINLENS_LLM_API_KEY_FILE` | Alternative to key above | Path to a mounted file containing the API key |
| `TRAINLENS_LLM_TIMEOUT_SECONDS` | No | Request timeout in seconds; default: `120` |

For example, in **Bash**, set the public values and enter the secret without
putting it in shell history:

```bash
export TRAINLENS_LLM_BASE_URL="https://your-provider.example/v1"
export TRAINLENS_LLM_MODEL="your-provider-model-id"
read -r -s -p "Provider API key: " TRAINLENS_LLM_API_KEY; echo
export TRAINLENS_LLM_API_KEY
```

For **containers or managed deployments**, you can mount a secret file and set
`TRAINLENS_LLM_API_KEY_FILE=/run/secrets/trainlens_llm_key` instead of putting
its value in an environment variable. If both variables are set, the non-empty
`TRAINLENS_LLM_API_KEY` takes precedence. TrainLens reads the secret file only
when a provider is configured; missing, empty, multiline, or oversized key
files are rejected with an error that does not print the key.

For **Jupyter/Colab on any OS**, use `getpass()` in a cell; it keeps the secret
out of the notebook source:

```python
import os
from getpass import getpass

os.environ["TRAINLENS_LLM_BASE_URL"] = "https://your-provider.example/v1"
os.environ["TRAINLENS_LLM_MODEL"] = "your-provider-model-id"
os.environ["TRAINLENS_LLM_API_KEY"] = getpass("Provider API key: ")
```

Alternatively, copy [`.env.example`](.env.example) to a local `.env` file
and replace its placeholders. **TrainLens does not load `.env` automatically.**
To make the values available from a trusted `.env` file in Bash/Zsh, run
`set -a; source .env; set +a` **before** launching Python/Jupyter.
If you use local Ollama, omit the API-key line entirely from that file.
The `.env` file is excluded by [`.gitignore`](.gitignore): never commit API
keys, paste them into tracked notebooks, or print them in logs. Environment
changes made after starting Jupyter may not reach an existing kernel; configure
the variables in the kernel with `os.environ` or restart it.

#### 3. Load a prompt and preview the LLM call

TrainLens includes built-in prompts and supports your own UTF-8 Markdown
`.md` prompt files. `load_prompt()` **reads the file as text**; pass that
text to `PromptOptions(objective=...)` to use it in a report. Prompts are not
automatically loaded from a directory or sent just because a file exists.

```python
from trainlens import (
    PromptOptions,
    build_improvement_ideas,
    load_prompt,
    preview_llm_request,
    show_trainlens_prompts,
)

# Use the training_state dictionary from Quick start.
# This example path exists when you clone the repository; otherwise create
# your own .md prompt file and pass its local path here.
objective = load_prompt("examples/prompts/improvement_plan.md")
options = PromptOptions(prompt_name="improvement_plan", objective=objective)

print([item.name for item in show_trainlens_prompts()])
preview = preview_llm_request(
    training_state,
    mode="improvement_ideas",
    prompt_options=options,
)
print(preview.endpoint, preview.model)
print(preview.system_prompt)
print(preview.user_prompt)  # Review evidence BEFORE sending anything

ideas = build_improvement_ideas(training_state, prompt_options=options)
print(ideas.markdown)  # Makes the actual LLM request
```

`preview_llm_request()` does **not** call the network;
`preview.payload()` shows the exact JSON request body without the API key.
You can skip the custom prompt and generate a standard scientific report:

```python
from trainlens import build_llm_report

report = build_llm_report(training_state)
print(report.markdown)
```

If authentication fails (HTTP 401/403), check the key and endpoint. For a
connection error, check that the local server is running and the base URL is
reachable; for model-related errors, check the model ID. Outbound reports send
bounded, best-effort-redacted training evidence to your chosen provider, so
inspect the preview and review its privacy terms first.

For provider injection, more prompt recipes, and privacy details, see
[LLM setup, prompts, and request previews](docs/en/llm-and-privacy.md)
or [la guía en español](docs/es/llm-y-privacidad.md).

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
