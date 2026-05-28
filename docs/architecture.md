# Architecture

TrainLens is organized around a narrow pipeline:

```text
Notebook Hook Layer
  -> Notebook Introspection Engine
  -> Training Session Analyzer
  -> Metric Interpretation Engine
  -> Explanation Generator
  -> Notebook Renderer
```

## Notebook Hook Layer

IPython magics expose the notebook UX. The extension should stay thin and delegate work to pure Python services.

## Introspection Engine

The inspector scans user namespace values, records compact variable metadata, and identifies model-like objects without importing large ML frameworks.

## Analyzer Layer

Analyzers receive a `NotebookSnapshot` and the most likely `ModelCandidate`. The default analyzer looks for metrics, dataset labels, feature names, and model attributes.

## Renderer Layer

Renderers convert `AnalysisResult` objects into Markdown or Rich output. They should not perform analysis.

## LLM Layer

LLM providers are optional. Providers receive a completed local report and may improve wording, but analyzers remain the source of evidence.
