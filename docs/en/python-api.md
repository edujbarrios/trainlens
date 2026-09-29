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

## Inspect training configuration

```python
from trainlens import inspect_training_profile

profile = inspect_training_profile(model, trainer=trainer, namespace=globals())
print(profile.strategy)
print(profile.trainable_components)
print(profile.frozen_components)
print(profile.parameters)
```

`inspect_training_profile()` uses safe duck typing and does not import optional
framework packages. It normalizes common Hugging Face training arguments,
PEFT/LoRA/DoRA settings, and VLM configuration such as projector type,
multimodal learning rates, visual feature selection, and sequence length.

For VLMs, TrainLens distinguishes projector-only alignment, adapter-based
fine-tuning, and broader VLM fine-tuning when the available object state is
sufficient. The profile also records detected trainable and frozen vision,
language, and multimodal-projector components. Inspection is best-effort: an
unknown or inaccessible attribute is omitted rather than making notebook
analysis fail.

The normal `%explain_training` and report workflows use the same profile to add
VLM-specific evidence and recommendations. The exact sanitized profile included
in an optional LLM request is visible through `preview_notebook_context()` or
`%explain_training --dry-run`.

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
comparison helpers, experiment planning, monitoring/callback classes,
`TrainingProfile`, `inspect_training_profile()`, and export helpers. Framework
inspection internals and model dataclasses under subpackages should be treated
as lower-level APIs.

