# Python API

## LLM reports

```python
from trainlens import build_improvement_ideas, build_paper_report

paper = build_paper_report(globals(), max_metric_points=12)
ideas = build_improvement_ideas(globals(), max_metric_points=12)
print(paper.markdown)
```

`build_llm_report` is an alias-style entry point for a paper report. A
`LiveReport` contains the generated Markdown and the extracted `AnalysisResult`.
`max_metric_points` must be at least 2; longer histories retain endpoints,
extrema, count, and an ordered sample.

## Compare runs

```python
from trainlens import compare_runs, render_run_comparison

result = compare_runs(
    {"loss": 0.40, "f1": 0.71},
    {"loss": 0.34, "f1": 0.75},
    baseline_name="run-a",
    experiment_name="run-b",
)
print(render_run_comparison(result))
```

Loss, error, perplexity, WER, CER, latency, FAD, and Fréchet-like names are
treated as lower-is-better. Accuracy, AUC, F1, precision, recall, score, mAP,
and NDCG-like names are higher-is-better. Unknown metric names receive deltas
without a success claim. Missing and non-finite values are handled explicitly.

## Plan one controlled experiment

```python
from trainlens import ExperimentRun, experiment_config, suggest_next_experiment

runs = [ExperimentRun(
    name="baseline",
    metrics={"train_loss": 0.20, "validation_loss": 0.50},
    parameters={"learning_rate": 1e-3, "dropout": 0.10, "batch_size": 32},
    estimated_cost="medium",
)]

recommendation = suggest_next_experiment(
    runs, objective_metric="validation_loss", minimum_improvement=0.01
)
next_parameters = experiment_config(
    recommendation, base_parameters=runs[0].parameters
)
```

The recommendation changes exactly one parameter and includes evidence,
confidence, cost, constants, and a measurable success criterion. It is a
deterministic heuristic, not a guarantee of improvement. Pass an objective with
a recognizable direction if automatic selection is unsuitable.

## Public entry points

The stable top-level package exports report builders, prompt configuration,
comparison helpers, experiment planning, monitoring/callback classes, and
export helpers. Framework inspection internals and model dataclasses under
subpackages should be treated as lower-level APIs.

