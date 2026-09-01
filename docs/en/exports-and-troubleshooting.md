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
can be passed explicitly. PDF requires `trainlens[pdf]`. The built-in HTML and
PDF renderers favor portable, dependency-light reports over full CommonMark or
typographic fidelity. JSON converts non-finite floating values to `null`.

## Common problems

### No active IPython shell

Pass a namespace: `build_paper_report(vars(my_module))` or a purpose-built
mapping. Automatic namespace lookup only works inside IPython/Jupyter.

### LLM provider is not configured

Set non-empty `TRAINLENS_LLM_BASE_URL`, `TRAINLENS_LLM_API_KEY`, and
`TRAINLENS_LLM_MODEL` values. Restart or re-run configuration cells if the
kernel was created earlier.

### Metrics are missing

Keep histories or trainer objects in the inspected namespace. Use conventional
names, numeric values, and finite values. TrainLens does not execute training or
retrieve metrics from a remote tracking server.

### Comparison direction is unknown

The metric name is outside the built-in vocabulary. The numeric delta is still
valid, but decide the desired direction in domain-specific code.

### PDF export fails

Install `python -m pip install "trainlens[pdf]"`. For complex publication-ready
layout, export Markdown or JSON and render it with a dedicated publishing tool.

## Current limitations

- Alpha-stage API and a Python 3.11+ requirement
- In-memory magic run history, with no persistent backend or collaboration UI
- Heuristic framework detection and metric direction inference
- Three focused live detectors rather than statistical drift analysis
- Deterministic next-step rules rather than causal or hyperparameter search
- LLM output quality and cost depend on the configured provider

These constraints make TrainLens most useful as a lightweight notebook aid and
reporting layer, not as an all-in-one MLOps platform.

