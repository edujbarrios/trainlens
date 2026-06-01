# Visual Explanations

TrainLens visual explanations are dark-mode SVG reports generated from the same `AnalysisResult` used by the Markdown renderer. They are dependency-free, notebook-friendly, and safe to commit as docs assets.

## Views

- `overview`: model, framework, counts for signals, metrics, trace events, and next steps.
- `metric_trace`: line chart from numeric values captured in execution trace events.
- `signal_panel`: compact severity view for warnings and findings.
- `feature_lens`: ranked evidence view from `top_features` or fallback training evidence.

## Generate Assets

```python
from trainlens.pipeline import explain_namespace
from trainlens.renderers.visual import DarkVisualRenderer


result = explain_namespace(globals())
assets = DarkVisualRenderer().write_dashboard_assets(result, "trainlens_assets")
```

The renderer always uses TrainLens dark styling. The intended direction is a visual training report that stands on its own: metrics explain what moved, traces explain when it moved, and signals explain why it matters.
