# TrainLens

TrainLens explains machine-learning training results from the full context of a
Jupyter notebook: model objects, dataset details, metric history, logs, traces,
hyperparameters, and experiment notes already present in memory.

[![CI](https://github.com/edujbarrios/trainlens/actions/workflows/ci.yml/badge.svg)](https://github.com/edujbarrios/trainlens/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)

Maintained by Eduardo J. Barrios.

TrainLens is built for AI/LLM-powered training analysis without leaving the
notebook. It produces a local Markdown diagnosis immediately, then can enhance
that same report with an OpenAI-compatible LLM provider from inside the same
notebook cell.

## Quickstart

Use TrainLens after or during a training run. Keep the training context in the
notebook namespace so the report can explain results against the dataset,
metrics, model setup, and run notes.

```python
import os

from IPython.display import Markdown, display

from trainlens.llm.enhancer import maybe_enhance
from trainlens.notebook import display_live_report


history = {
    "train_loss": [2.4, 1.9, 1.6],
    "eval_loss": [2.5, 2.0, 1.8],
}
dataset_name = "customer-support-instructions"
dataset_notes = "Small validation split; long answers are underrepresented."
model_name = "mistral-lora-support-bot"
learning_rate = 2e-5

report = display_live_report(globals())

os.environ["TRAINLENS_LLM_BASE_URL"] = "https://api.example.com/v1"
os.environ["TRAINLENS_LLM_API_KEY"] = "your-api-key"
os.environ["TRAINLENS_LLM_MODEL"] = "your-model-name"

enhanced_report = maybe_enhance(report.markdown)
display(Markdown(enhanced_report))
```

The first `display_live_report(globals())` call works without an API key. The
LLM enhancement uses the environment variables shown above and stays inside the
notebook.

## What TrainLens Explains

TrainLens is focused on practical training questions:

- What did this notebook train?
- Which dataset and training context shaped the result?
- Which metrics matter right now?
- Is the run improving, plateauing, overfitting, or missing validation evidence?
- What should I try next in this same notebook?

It looks for evidence already available in the active notebook:

- model objects and framework hints
- dataset names, labels, feature names, and notes
- Keras histories, Hugging Face `log_history`, PyTorch loop metrics, and Lightning-style metrics
- execution traces such as `training_trace`, `execution_trace`, `logs`, or `trainer.state.log_history`
- loss, perplexity, recall@k, contrastive loss, projector loss, audio reconstruction loss, and eval metrics
- LoRA/adapters, trainable parameter ratios, multimodal tower/projector hints, and audio-conditioning clues

TrainLens intentionally does not provide a GUI, image dashboard, or screenshot
workflow. The main interface is notebook Markdown.

## How It Works

```text
Notebook variables
  -> namespace snapshot
  -> model, dataset, and framework detection
  -> metric and execution-trace extraction
  -> training heuristics
  -> notebook Markdown report
  -> optional LLM-enhanced explanation
```

Core modules live under `src/trainlens`:

- `introspection`: notebook namespace scanning and model discovery
- `analyzers`: framework and training-session analyzers
- `heuristics`: loss, convergence, contrastive, adapter, projector, multimodal, and music-generation rules
- `models`: typed domain objects
- `llm`: OpenAI-compatible report enhancement
- `magic`: IPython magic commands
- `renderers`: notebook Markdown and Rich output
- `storage`: notebook-local run persistence and comparison

## Notebook Usage

Load TrainLens from a clone without installing it into the environment:

```python
from pathlib import Path
import sys

TRAINLENS_REPO = Path("/path/to/trainlens").resolve()
sys.path.insert(0, str(TRAINLENS_REPO / "src"))
```

Render the current notebook report:

```python
from trainlens.notebook import display_live_report

live_report = display_live_report(globals())
```

Use IPython magics:

```python
%load_ext trainlens.magic.extension
%explain_training
%training_summary
%why_bad_model
%compare_runs
```

See [docs/live-notebook-cells.md](docs/live-notebook-cells.md) for a compact
copy-paste workflow, and [docs/report-sections.md](docs/report-sections.md) for
how to read each report section.

## Optional LLM Provider

TrainLens uses OpenAI-compatible chat completions for report enhancement. Set
these variables in the notebook or in the process environment:

```python
import os

os.environ["TRAINLENS_LLM_BASE_URL"] = "https://api.example.com/v1"
os.environ["TRAINLENS_LLM_API_KEY"] = "your-api-key"
os.environ["TRAINLENS_LLM_MODEL"] = "your-model-name"
```

Then enhance any generated report:

```python
from IPython.display import Markdown, display
from trainlens.llm.enhancer import maybe_enhance

display(Markdown(maybe_enhance(live_report.markdown)))
```

No API key is required for the local report. API access is only used when you
call the LLM enhancement path.

## Examples

Run a no-PyTorch dependency example:

```bash
python examples/pytorch_loop_metrics.py
```

More examples live in [notebooks/](notebooks/).

## Roadmap

- richer Hugging Face Trainer, Accelerate, PEFT, and DeepSpeed history extraction
- first-class CLIP, SigLIP, ViT, projector, and VLM validation reports
- first-class AI music generation diagnostics for audio tokenizers, diffusion models, CLAP alignment, MIDI/symbolic models, vocals, stems, and long-form structure
- notebook cell execution hooks
- plugin discovery through entry points
- local model adapters

## Contributing

TrainLens is open source from day one. Contributions are welcome across
analyzers, notebook UX, docs, examples, and tests. See
[CONTRIBUTING.md](CONTRIBUTING.md) and [docs/roadmap.md](docs/roadmap.md).

## License

Apache License 2.0. See [LICENSE](LICENSE).
