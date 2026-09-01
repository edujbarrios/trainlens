# TrainLens documentation

TrainLens inspects training evidence already available in Python or a Jupyter
notebook and turns it into reviewable reports, comparisons, alerts, and
controlled follow-up experiments. Most analysis is local and deterministic;
LLM-generated prose is optional.

## Start here

- [Installation and quickstart](getting-started.md)
- [Notebook workflow and framework adapters](notebooks.md)
- [Python API: reports, comparisons, and experiments](python-api.md)
- [Monitoring and callbacks](monitoring.md)
- [LLM configuration, prompts, and privacy](llm-and-privacy.md)
- [Exports, limitations, and troubleshooting](exports-and-troubleshooting.md)

## When TrainLens is useful

TrainLens is a good fit when experiment state lives in notebook variables and
you want a lightweight, framework-neutral record without deploying a tracking
server. It is especially useful for comparing a small number of runs, catching
simple metric anomalies during training, and creating a consistent report for
review.

It is not a replacement for a full experiment tracker, profiler, model
evaluator, or causal diagnosis system. Its heuristics are intentionally small
and explainable. Always verify generated conclusions against your data and
domain knowledge.

## Requirements

- Python 3.11 or newer
- IPython/Jupyter for magic commands
- An OpenAI-compatible endpoint only for LLM-generated reports
- `reportlab` only for PDF export

