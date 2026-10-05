# TrainLens documentation

TrainLens inspects training and dataset evidence already available in Python or a Jupyter
notebook and turns it into reviewable reports, comparisons, alerts, and controlled follow-up
experiments. Most analysis is local and deterministic; LLM-generated output is optional.

TrainLens 0.16 also exposes a provider-free Agent Mode so AutoResearch-style skills and coding
agents can use TrainLens as a deterministic evidence layer without configuring an API key inside
TrainLens.

## Start here

- [Installation and quickstart](getting-started.md)
- [Agent Mode and AutoResearch-style skills](agent-mode.md)
- [Dataset-aware analysis, including image and multimodal datasets](dataset-context.md)
- [Experiment history, repeated runs, checkpoints, provenance, and CI](experiment-workflows.md)
- [Notebook workflow and framework adapters](notebooks.md)
- [Python API: reports, comparisons, and experiments](python-api.md)
- [Monitoring and callbacks](monitoring.md)
- [Analyzer plugins](analyzer-plugins.md)
- [LLM configuration, prompts, and privacy](llm-and-privacy.md)
- [Verified LLM improvement plans](verified-llm-plans.md)
- [Exports, limitations, and troubleshooting](exports-and-troubleshooting.md)

## When TrainLens is useful

TrainLens is a good fit when experiment state lives in notebook variables or portable run files
and you want a lightweight, framework-neutral evidence layer without deploying a tracking server.
It can explain aggregate tabular, text, image, and image-centered multimodal dataset context;
persist a small local run history; compare objectives and constraints; reason cautiously about
repeated seeds; capture reproducibility evidence; and turn observations into controlled next runs.

It is not a replacement for a full experiment tracker, profiler, model evaluator, computer
vision pipeline, AutoML engine, or causal diagnosis system. Its deterministic heuristics are
intentionally small and explainable. Always verify generated or agent-proposed conclusions against
your data and domain knowledge.

## Requirements

- Python 3.11–3.14
- IPython/Jupyter for magic commands
- no model provider or API key for Agent Mode
- an OpenAI-compatible endpoint only for optional direct LLM-generated reports
- `matplotlib` only for the `plots` extra
- `reportlab` only for PDF export
