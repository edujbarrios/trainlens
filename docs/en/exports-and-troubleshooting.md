# Exports, limitations, and troubleshooting

## Export a report

```python
from trainlens import render_report, write_report

markdown = render_report(report, format="markdown")
write_report(report, "report.md")
write_report(report, "report.html")
write_report(report, "report.json")
write_report(report, "report.pdf")
```

The format is inferred from `.md`, `.markdown`, `.html`, `.json`, or `.pdf`, or
can be passed explicitly. PDF works after plain `pip install trainlens`.
The built-in HTML and PDF renderers favor portable reports over full CommonMark or
typographic fidelity. JSON converts non-finite floating values to `null`.

## Common problems

### No active IPython shell

Pass a namespace: `build_paper_report(vars(my_module))` or a purpose-built
mapping. Automatic namespace lookup only works inside IPython/Jupyter.

### LLM provider is not configured

Set non-empty `TRAINLENS_LLM_BASE_URL` and `TRAINLENS_LLM_MODEL` values. Set
`TRAINLENS_LLM_API_KEY` when the endpoint requires authentication; it may be
omitted for local unauthenticated servers. Restart or re-run configuration cells
if the kernel was created earlier.

### Metrics are missing

Keep histories or trainer objects in the inspected namespace, or provide
explicit metrics through `AnalysisConfig(metrics=...)`. Use numeric, finite
values. TrainLens does not execute training or retrieve metrics from a remote
tracking server.

### Comparison direction is unknown

Register domain-specific semantics with `register_metric(MetricSpec(...))` when
a metric name is outside the built-in vocabulary. Without a registered
direction, the numeric delta remains valid but TrainLens will not label it as an
improvement or regression.

### LLM context was truncated

`ContextPolicy` bounds outbound evidence. Use `preview_notebook_context()` to
inspect the exact sanitized context and increase only the specific budgets you
need. TrainLens marks omitted items and global character truncation explicitly.

### PDF export fails

The standard install includes ReportLab. If it is missing (for example after
using `--no-deps`), reinstall with `python -m pip install --upgrade --force-reinstall trainlens`.
For complex publication-ready
layout, export Markdown or JSON and render it with a dedicated publishing tool.

## Current limitations

- Alpha-stage API and supported Python 3.11–3.14 (not 3.15)
- Notebook magic history remains in-memory, although completed `TrainingRun`
  artifacts can now be saved/loaded as JSON
- Heuristic framework detection; ambiguous notebooks should use `AnalysisConfig`
  for explicit model/trainer selection
- A small built-in live-detector set; project-specific detectors can be added
  through the public `AlertDetector` interface
- Deterministic next-step and Pareto heuristics rather than causal inference or
  full hyperparameter search
- LLM output quality, privacy terms, and cost depend on the configured or
  injected provider

These constraints make TrainLens most useful as a lightweight notebook aid and
reporting layer, not as an all-in-one MLOps platform.
