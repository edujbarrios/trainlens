# Architecture

TrainLens is organized around a narrow pipeline:

```text
Notebook Hook Layer
  -> Notebook Introspection Engine
  -> Training Session Analyzer
  -> Execution Trace Extractor
  -> Metric Interpretation Engine
  -> Explanation Generator
  -> Notebook and Visual Renderers
```

## Notebook Hook Layer

IPython magics expose the notebook UX. The extension should stay thin and delegate work to pure Python services.

## Introspection Engine

The inspector scans user namespace values, records compact variable metadata, and identifies model-like objects without importing large ML frameworks.

## Analyzer Layer

Analyzers receive a `NotebookSnapshot` and the most likely `ModelCandidate`. The default analyzer looks for metrics, dataset labels, feature names, and model attributes.

Metric extraction normalizes common notebook training artifacts without importing
heavy ML frameworks. Supported shapes include Keras-style `history` mappings,
Hugging Face `log_history`, PyTorch loop lists such as `train_losses` and
`val_losses`, epoch dictionaries, and Lightning-style `callback_metrics` or
`logged_metrics`.

Execution traces are extracted separately from variables such as
`training_trace`, `execution_trace`, `logs`, `log_history`, and
`trainer.state.log_history`. Trace events can feed both Markdown tables and
dark-mode visual explanations.

## Renderer Layer

Renderers convert `AnalysisResult` objects into Markdown, Rich output, or
dependency-free SVG visuals. They should not perform analysis.

The visual renderer writes dark-mode SVG assets for overview cards, metric
traces, signal panels, and ranked evidence views. These visuals use the same
analysis output as the notebook Markdown report, so docs and notebooks stay
aligned.

## LLM Layer

LLM providers are optional. Providers receive a completed local report and may improve wording, but analyzers remain the source of evidence.

Prompting is handled by a parameterized Jinja2 template in `trainlens.llm.prompts`.
The template receives the local report, audience, tone, model-family context,
rules, and focus areas so provider adapters do not hardcode prompt strings.
