# TrainLens

TrainLens is an open-source framework for understanding machine learning training sessions directly inside Jupyter notebooks.

[![CI](https://github.com/edujbarrios/trainlens/actions/workflows/ci.yml/badge.svg)](https://github.com/edujbarrios/trainlens/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)

TrainLens inspects notebook state, training histories, model objects, metrics, and prediction behavior to explain how a model is training and what to try next. It works locally by default with heuristic analysis and can optionally enhance explanations through an LLM provider.

No package install is required for the direct llm7.io workflow. Clone the repo, point the tool at a Markdown report, and it will call llm7.io using only the Python standard library.

```text
Model detected: RandomForestClassifier

Training summary:
- Accuracy improved from 0.81 to 0.89
- Validation performance stabilized
- Slight overfitting detected

Potential issues:
- Class imbalance
- Small validation split

Top features:
1. age
2. account_balance
3. transaction_count

Recommended next steps:
- Tune max_depth
- Try stratified split
- Inspect false negatives
```

## Why TrainLens?

Most notebook explainability tools require a dedicated explainability package or a lot of manual wiring. TrainLens starts with what your notebook already has:

- Python variables in the active IPython shell
- trained model objects
- metric dictionaries and training histories
- dataset-like objects and feature names
- prediction arrays and validation targets

It then turns that evidence into useful summaries, debugging signals, recommendations, and experiment ideas.

## Use Without Installing

```bash
python tools/trainlens_llm7.py --help
```

TrainLens is designed so the notebook-side explainer can run locally without requiring users to install an extra Python library. The standalone llm7.io tool uses only the Python standard library and reads configuration from environment variables.

```env
TRAINLENS_LLM_BASE_URL=https://api.llm7.io/v1
TRAINLENS_LLM_API_KEY=your_llm7_api_key_here
TRAINLENS_LLM_MODEL=auto
```

```bash
python tools/trainlens_llm7.py training_report.md
```

From a notebook cell, without installing TrainLens as a library:

```python
!python tools/trainlens_llm7.py training_report.md
```

Contributor setup for package internals lives in [CONTRIBUTING.md](CONTRIBUTING.md).

## Optional LLM Enhancement

TrainLens never requires API access. LLM support is opt-in and provider based.

```env
TRAINLENS_LLM_BASE_URL=https://api.llm7.io/v1
TRAINLENS_LLM_API_KEY=your_llm7_api_key_here
TRAINLENS_LLM_MODEL=auto
```

The current placeholder provider targets llm7.io-compatible chat completions. The provider interface is intentionally small so future adapters can support OpenAI, Anthropic, Ollama, and local models without changing analyzer logic.

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
- `heuristics`: overfitting, convergence, class-balance, and metric rules
- `models`: typed domain objects
- `llm`: provider abstractions and optional enhancement
- `magic`: IPython magic commands
- `renderers`: notebook Markdown/Rich output
- `storage`: run persistence and comparison

## Roadmap

- richer PyTorch and TensorFlow history extraction
- first-class xgboost/lightgbm feature importance support
- notebook cell execution hooks
- plugin discovery through entry points
- local model adapters
- visual diagnostics for metric drift
- public example notebooks and screenshots

## Screenshots

Screenshots and short demos will live in `docs/assets/` as the notebook UI stabilizes.

## Contributing

TrainLens is open source from day one. Contributors are welcome across analyzers, notebook UX, docs, examples, and tests. See [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/roadmap.md](docs/roadmap.md).

## License

Apache License 2.0. See [LICENSE](LICENSE).
