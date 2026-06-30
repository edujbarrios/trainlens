# TrainLens

TrainLens explains training results from the full context of your Jupyter
notebook: the model, dataset, metrics, logs, traces, hyperparameters, and notes
you already have in memory.

It is built for AI/LLM-powered training analysis without leaving the notebook.
Local heuristics work immediately, and an OpenAI-compatible LLM can enhance the
same report in-place when API credentials are configured.

```python
# In a Jupyter notebook, after or during training:
from IPython.display import Markdown, display

from trainlens.llm.enhancer import maybe_enhance
from trainlens.notebook import display_live_report


history = {
    "train_loss": [2.4, 1.9, 1.6],
    "eval_loss": [2.5, 2.0, 1.8],
}
dataset_name = "customer-support-instructions"
dataset_notes = "Small validation split; long answers are underrepresented."
learning_rate = 2e-5

report = display_live_report(globals())

# Ask the LLM to explain the results using the notebook context.
# First set TRAINLENS_LLM_BASE_URL, TRAINLENS_LLM_API_KEY, and TRAINLENS_LLM_MODEL.
display(Markdown(maybe_enhance(report.markdown)))
```

[![CI](https://github.com/edujbarrios/trainlens/actions/workflows/ci.yml/badge.svg)](https://github.com/edujbarrios/trainlens/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)

TrainLens reads the variables that already exist in a notebook, finds training metrics, dataset context, and run evidence, and renders a concise Markdown explanation in the notebook output. It is focused on answering practical questions after or during a training run:

- What did this notebook train?
- What dataset and training context shaped the result?
- Which metrics matter right now?
- Is the run improving, plateauing, overfitting, or missing validation evidence?
- What should I try next in this same notebook?

The project intentionally does not provide a GUI, image dashboard, or screenshot workflow. The primary interface is notebook Markdown.

## What TrainLens Does

TrainLens gives you a notebook-local training report:

1. It scans the notebook namespace for models, histories, metrics, traces, and useful metadata.
2. It normalizes common training artifacts such as Keras histories, Hugging Face `log_history`, PyTorch loop metrics, and Lightning-style metric dictionaries.
3. It applies heuristics for loss behavior, validation gaps, class balance, adapters, projectors, contrastive training, multimodal runs, and music-generation experiments.
4. It renders a Markdown report with a training summary, result explanation, potential issues, recommended next steps, and an improvement plan.
5. Optionally, it sends that Markdown report to an OpenAI-compatible chat-completions endpoint for LLM-enhanced wording.

No API key is required for local analysis. API access is only needed for optional LLM enhancement.

## Why TrainLens?

Most notebook training reports require manual plotting, custom logging, or a separate experiment-tracking service. TrainLens starts with what your notebook already has:

- Python variables in the active IPython shell
- trained model objects from Transformers, PEFT, PyTorch, timm, diffusers, audio stacks, and notebook code
- metric dictionaries and trainer histories from LLM/VLM fine-tuning loops
- execution traces such as `training_trace`, `execution_trace`, `log_history`, or `trainer.state.log_history`
- loss, perplexity, recall@k, contrastive loss, projector loss, audio reconstruction loss, and eval metrics
- LoRA/adapters, trainable parameter ratios, multimodal tower/projector hints, and audio-conditioning clues

It then turns that evidence into a readable notebook report.

## How It Works

```text
Notebook variables
  -> namespace snapshot
  -> model and framework detection
  -> metric and execution-trace extraction
  -> training heuristics
  -> notebook Markdown report
  -> optional LLM enhancement
```

You can use TrainLens in two notebook-first ways:

- Call `display_live_report(globals())` from a notebook cell.
- Load the IPython extension and run `%explain_training`, `%training_summary`, `%why_bad_model`, or `%compare_runs`.

You can also pass a saved Markdown report to `tools/trainlens_openai_compatible.py` when you want optional LLM enhancement from a notebook shell cell.

## Use From A Clone

Use TrainLens directly from a cloned repository by adding `src/` to the notebook path. This does not install a package into the environment.

```python
from pathlib import Path
import sys

TRAINLENS_REPO = Path("/path/to/trainlens").resolve()
sys.path.insert(0, str(TRAINLENS_REPO / "src"))
```

## Notebook Workflow

Cell 1: keep your normal training variables in the notebook.

```python
model = my_model
train_losses = [2.3, 1.9, 1.55, 1.34]
val_losses = [2.4, 2.0, 1.78, 1.72]
epoch_logs = [
    {"epoch": 1, "train_loss": 2.3, "val_loss": 2.4},
    {"epoch": 2, "train_loss": 1.9, "val_loss": 2.0},
    {"epoch": 3, "train_loss": 1.55, "val_loss": 1.78},
    {"epoch": 4, "train_loss": 1.34, "val_loss": 1.72},
]
```

Cell 2: render the report inside the notebook.

```python
from trainlens.notebook import display_live_report

live_report = display_live_report(globals())
```

`live_report.result` keeps the structured analysis object, while `live_report.markdown` keeps the rendered notebook report.

The Markdown report includes:

- `Training summary`: what TrainLens found in the notebook state.
- `Result explanation`: what the metrics and signals mean.
- `Potential issues`: concrete risks detected by heuristics.
- `Recommended next steps`: direct suggestions.
- `Improvement plan`: prioritized actions with confidence and rationale.

Cell 3: use notebook magics after loading the extension.

```python
%load_ext trainlens.magic.extension
%explain_training
%training_summary
%why_bad_model
%compare_runs
```

See [docs/live-notebook-cells.md](docs/live-notebook-cells.md) for a compact copy-paste workflow, and [docs/report-sections.md](docs/report-sections.md) for how to read each report section.

## Analyze An LLM Fine-Tune

```python
from IPython.display import Markdown, display
from trainlens.pipeline import explain_namespace
from trainlens.renderers.markdown import MarkdownRenderer


class MistralLoRAFineTune:
    class Config:
        model_type = "mistral"

    config = Config()


model = MistralLoRAFineTune()
history = {
    "train_loss": [2.6, 2.1, 1.92, 1.91],
    "eval_loss": [2.5, 2.18, 2.16, 2.16],
    "eval_perplexity": [12.2, 9.1, 8.9, 8.9],
}
lora_rank = 4
trainable_params = 8_000_000
total_params = 7_000_000_000

report = MarkdownRenderer().render(explain_namespace(globals()))
display(Markdown(report))
```

## Include Execution Traces

TrainLens can include recent training events in the report. Use a list of dictionaries named `training_trace`, `execution_trace`, `trace`, `logs`, or `log_history`. Hugging Face-style `trainer.state.log_history` is also detected.

```python
training_trace = [
    {"step": 100, "epoch": 0.5, "event": "batch_end", "loss": 1.92, "lr": 2e-5},
    {"step": 200, "epoch": 1.0, "event": "eval", "eval_loss": 1.76, "eval_perplexity": 5.8},
    {"step": 300, "epoch": 1.5, "event": "checkpoint", "message": "saved adapter weights"},
]

report = MarkdownRenderer().render(explain_namespace(globals()))
display(Markdown(report))
```

The report will show an `Execution trace` table with step, epoch, event, and numeric metrics. This makes it easier to connect the final diagnosis with what actually happened during the run.

## Analyze PyTorch Loop Metrics

TrainLens detects common PyTorch training-loop variables such as `train_losses`, `val_losses`, `epoch_logs`, `callback_metrics`, and `logged_metrics`. Tensor-like scalar values are supported when they expose `.item()`, which covers the common `torch.Tensor` case.

```python
train_losses = [2.3, 1.9, 1.55, 1.34]
val_losses = [2.4, 2.0, 1.78, 1.72]
epoch_logs = [
    {"epoch": 1, "train_loss": 2.3, "val_loss": 2.4},
    {"epoch": 2, "train_loss": 1.9, "val_loss": 2.0},
]

display(Markdown(MarkdownRenderer().render(explain_namespace(globals()))))
```

For a runnable no-PyTorch dependency example:

```bash
python examples/pytorch_loop_metrics.py
```

## Optional LLM Enhancement

TrainLens never requires API access. LLM support is opt-in and provider based.

```env
TRAINLENS_LLM_BASE_URL=https://api.example.com/v1
TRAINLENS_LLM_API_KEY=your_api_key_here
TRAINLENS_LLM_MODEL=auto
```

Generate a report, save it as Markdown, preview it in the notebook, and then send that same file to an OpenAI-compatible endpoint:

```python
from pathlib import Path

from IPython.display import Markdown, display
from trainlens.pipeline import explain_namespace
from trainlens.renderers.markdown import MarkdownRenderer


report_path = Path("training_report.md")
report = MarkdownRenderer().render(explain_namespace(globals()))
report_path.write_text(report, encoding="utf-8")

display(Markdown(report))
print(f"Wrote {report_path.resolve()}")
```

```python
!python /path/to/trainlens/tools/trainlens_openai_compatible.py training_report.md
```

The current provider targets OpenAI-compatible chat completions. The provider interface is intentionally small so hosted APIs, Ollama-compatible gateways, and local model servers can be swapped without changing analyzer logic.

## Architecture

```text
Notebook Hook Layer
  -> Notebook Introspection Engine
  -> Training Session Analyzer
  -> Execution Trace Extractor
  -> Metric Interpretation Engine
  -> Explanation Generator
  -> Notebook Markdown Renderer
```

Core modules live under `src/trainlens`:

- `introspection`: notebook namespace scanning and model discovery
- `analyzers`: framework and training-session analyzers
- `heuristics`: loss, convergence, contrastive, adapter, projector, multimodal, and music-generation rules
- `models`: typed domain objects
- `llm`: provider abstractions and optional enhancement
- `magic`: IPython magic commands
- `renderers`: notebook Markdown and Rich output
- `storage`: notebook-local run persistence and comparison

More examples live in [notebooks/](notebooks/). Contributor setup for package internals lives in [CONTRIBUTING.md](CONTRIBUTING.md).

## Roadmap

- richer Hugging Face Trainer, Accelerate, PEFT, and DeepSpeed history extraction
- first-class CLIP, SigLIP, ViT, projector, and VLM validation reports
- first-class AI music generation diagnostics for audio tokenizers, diffusion models, CLAP alignment, MIDI/symbolic models, vocals, stems, and long-form structure
- notebook cell execution hooks
- plugin discovery through entry points
- local model adapters

## Contributing

TrainLens is open source from day one. Contributors are welcome across analyzers, notebook UX, docs, examples, and tests. See [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/roadmap.md](docs/roadmap.md).

## License

Apache License 2.0. See [LICENSE](LICENSE).
