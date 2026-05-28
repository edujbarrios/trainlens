# TrainLens

TrainLens is an open-source framework for understanding machine learning training sessions directly inside Jupyter notebooks.

[![CI](https://github.com/edujbarrios/trainlens/actions/workflows/ci.yml/badge.svg)](https://github.com/edujbarrios/trainlens/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)

TrainLens inspects notebook state, training histories, model objects, metrics, and prediction behavior to explain how a model is training and what to try next. The project is now focused on understanding foundation-model training and fine-tuning runs: LLMs, CLIP-style contrastive models, ViTs, multimodal projectors, and VLMs.

No package install is required for the direct OpenAI-compatible API workflow. Clone the repo, point the tool at a Markdown report, and it will call your configured chat-completions endpoint using only the Python standard library.

```text
Model detected: LlavaForConditionalGeneration

Training summary:
- Foundation-model profile: LLM, PROJECTOR, VLM
- Validation loss plateaued across recent checkpoints
- Tiny trainable parameter ratio detected

Potential issues:
- Projector alignment may be under-capacity
- Retrieval/caption validation is missing

Recommended next steps:
- Track eval_loss, perplexity, learning-rate schedule, and gradient norm
- Validate projector alignment with frozen vision and language towers
- Evaluate text-image retrieval or held-out multimodal instructions
```

## Why TrainLens?

Most notebook explainability tools require a dedicated explainability package or a lot of manual wiring. TrainLens starts with what your notebook already has:

- Python variables in the active IPython shell
- trained model objects from Transformers, PEFT, PyTorch, timm, diffusers, and notebook code
- metric dictionaries and trainer histories from LLM/VLM fine-tuning loops
- loss, perplexity, recall@k, contrastive loss, projector loss, and eval metrics
- LoRA/adapters, trainable parameter ratios, and multimodal tower/projector hints

It then turns that evidence into useful summaries, debugging signals, recommendations, and experiment ideas.

## Use Without Installing

```bash
git clone https://github.com/edujbarrios/trainlens.git
cd trainlens
python tools/trainlens_openai_compatible.py --help
```

TrainLens is designed so the notebook-side explainer can run locally without requiring users to install an extra Python library. The standalone OpenAI-compatible API tool uses only the Python standard library and reads configuration from environment variables.

```env
TRAINLENS_LLM_BASE_URL=https://api.example.com/v1
TRAINLENS_LLM_API_KEY=your_api_key_here
TRAINLENS_LLM_MODEL=auto
```

```bash
python tools/trainlens_openai_compatible.py training_report.md
```

From a notebook cell, without installing TrainLens as a library:

```python
!python tools/trainlens_openai_compatible.py training_report.md
```

## Jupyter Cell Examples

Use TrainLens directly from a cloned repo by adding `src/` to the notebook path.
This does not install a package into the environment.

```python
from pathlib import Path
import sys

TRAINLENS_REPO = Path("/path/to/trainlens").resolve()
sys.path.insert(0, str(TRAINLENS_REPO / "src"))
```

### Analyze an LLM fine-tune

```python
from trainlens.pipeline import explain_namespace
from trainlens.renderers.markdown import MarkdownRenderer
from IPython.display import Markdown, display


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

### Analyze a VLM projector run

```python
class LlavaProjectorRun:
    vision_tower = object()
    language_model = object()
    mm_projector = object()

    class Config:
        model_type = "llava"

    config = Config()


model = LlavaProjectorRun()
history = {
    "train_loss": [2.4, 1.9, 1.72, 1.71],
    "eval_loss": [2.3, 2.0, 1.99, 1.99],
}
lora_rank = 4

display(Markdown(MarkdownRenderer().render(explain_namespace(globals()))))
```

### Enhance a report with an OpenAI-compatible API from a notebook cell

```python
Path("training_report.md").write_text(report, encoding="utf-8")
```

```python
!python /path/to/trainlens/tools/trainlens_openai_compatible.py training_report.md
```

More examples live in [`notebooks/`](notebooks/).

Contributor setup for package internals lives in [CONTRIBUTING.md](CONTRIBUTING.md).

## Optional LLM Enhancement

TrainLens never requires API access. LLM support is opt-in and provider based.

```env
TRAINLENS_LLM_BASE_URL=https://api.example.com/v1
TRAINLENS_LLM_API_KEY=your_api_key_here
TRAINLENS_LLM_MODEL=auto
```

The current provider targets OpenAI-compatible chat completions. The provider interface is intentionally small so hosted APIs, Ollama-compatible gateways, and local model servers can be swapped without changing analyzer logic.

## Architecture

```text
Notebook Hook Layer
  -> Notebook Introspection Engine
  -> Training Session Analyzer
  -> Metric Interpretation Engine
  -> Explanation Generator
  -> Notebook Renderer
```

Core modules live under `src/trainlens`:

- `introspection`: notebook namespace scanning and model discovery
- `analyzers`: framework and training-session analyzers
- `heuristics`: loss, convergence, contrastive, adapter, projector, and multimodal rules
- `models`: typed domain objects
- `llm`: provider abstractions and optional enhancement
- `magic`: IPython magic commands
- `renderers`: notebook Markdown/Rich output
- `storage`: run persistence and comparison

## Roadmap

- richer Hugging Face Trainer, Accelerate, PEFT, and DeepSpeed history extraction
- first-class CLIP, SigLIP, ViT, projector, and VLM validation reports
- notebook cell execution hooks
- plugin discovery through entry points
- local model adapters
- visual diagnostics for fine-tuning drift and alignment regressions
- public example notebooks and screenshots

## Screenshots

Screenshots and short demos will live in `docs/assets/` as the notebook UI stabilizes.

## Contributing

TrainLens is open source from day one. Contributors are welcome across analyzers, notebook UX, docs, examples, and tests. See [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/roadmap.md](docs/roadmap.md).

## License

Apache License 2.0. See [LICENSE](LICENSE).
