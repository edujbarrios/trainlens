# TrainLens documentation

TrainLens inspects training evidence already available in Python or a Jupyter notebook and
turns it into reviewable reports, comparisons, alerts, and controlled follow-up experiments.
Most analysis is local and deterministic; LLM-generated output is optional.

## Start here

- [Installation and quickstart](getting-started.md)
- [Experiment history, repeated runs, checkpoints, provenance, and CI](experiment-workflows.md)
- [Notebook workflow and framework adapters](notebooks.md)
- [Python API: reports, comparisons, and experiments](python-api.md)
- [Monitoring and callbacks](monitoring.md)
- [Analyzer plugins](analyzer-plugins.md)
- [LLM configuration, prompts, and privacy](llm-and-privacy.md)
- [Verified LLM improvement plans](verified-llm-plans.md)
- [Exports, limitations, and troubleshooting](exports-and-troubleshooting.md)

## When TrainLens is useful

TrainLens is a good fit when experiment state lives in notebook variables and you want a
lightweight, framework-neutral record without deploying a tracking server. It can persist a
small local run history, compare objectives and constraints, reason cautiously about repeated
seeds, capture reproducibility evidence, and turn observations into controlled next runs.

It is not a replacement for a full experiment tracker, profiler, model evaluator, or causal
diagnosis system. Its deterministic heuristics are intentionally small and explainable. Always
verify generated conclusions against your data and domain knowledge.

## Requirements

- Python 3.11–3.14
- IPython/Jupyter for magic commands
- An OpenAI-compatible endpoint only for optional LLM-generated reports
- `matplotlib` only for the `plots` extra
- `reportlab` only for PDF export
