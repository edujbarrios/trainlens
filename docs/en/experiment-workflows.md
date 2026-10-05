# Experiment workflows

TrainLens can treat portable `TrainingRun` objects as a small local experiment history without
requiring a tracking server. The APIs in this guide are local, dependency-light, and designed
for notebook-scale research workflows.

## Keep runs in a local project

```python
from trainlens import Project

project = Project(".trainlens")
project.capture(
    globals(),
    name="lr-2e-5-seed-1",
    tags=("lora", "baseline"),
    parameters={"learning_rate": 2e-5, "seed": 1},
)

runs = project.runs()
```

`Project` stores the existing portable TrainLens run JSON format under the project directory
and keeps project labels such as names and tags in a separate local index.

## Rank several runs

```python
from trainlens import MetricConstraint, ObjectiveSpec, leaderboard

board = leaderboard(
    project.runs(),
    [
        ObjectiveSpec(metric="validation_loss", direction="min"),
        ObjectiveSpec(metric="f1", direction="max"),
    ],
    constraints=[MetricConstraint(metric="f1", operator=">=", threshold=0.80)],
)

board  # Markdown table in Jupyter
```

The leaderboard ranks by the primary objective, marks runs that satisfy hard constraints,
and identifies the Pareto-efficient subset across all objectives.

## Aggregate repeated seeds

Use grouping parameters to distinguish configurations while leaving the seed out of the group
key:

```python
from trainlens import aggregate_runs, compare_run_groups

groups = aggregate_runs(
    project.runs(),
    by=["learning_rate", "weight_decay"],
    metrics=["validation_loss", "f1"],
)

comparison = compare_run_groups(groups[0], groups[1], metric="f1")
print(comparison.delta, comparison.pooled_stdev, comparison.confidence)
```

TrainLens reports count, mean, sample standard deviation, minimum, and maximum. Group
comparisons relate the observed mean difference to between-run spread and deliberately avoid
claiming formal statistical significance from small seed counts.

## Learn from controlled parameter changes

```python
from trainlens import parameter_effects

effects = parameter_effects(project.runs(), metric="validation_loss")
for effect in effects:
    print(effect.parameter, effect.from_value, effect.to_value)
    print(effect.median_improvement, effect.improvement_rate, effect.confidence)
```

Only pairs that differ in exactly one recorded parameter are used. The result keeps the
concrete run-pair evidence so an observed effect can be inspected rather than treated as a
causal claim.

## Select a checkpoint without tuning on the test split

```python
from trainlens import select_checkpoint

selection = select_checkpoint(
    project.runs()[0],
    metric="validation_loss",
    min_delta=0.001,
)

print(selection.best_step)
print(selection.best_value)
print(selection.suggested_patience)
```

Optimization direction is inferred from TrainLens metric semantics or can be supplied for a
custom metric. Held-out test metrics are rejected as checkpoint-selection targets.

## Capture provenance

```python
from trainlens import capture_provenance

provenance = capture_provenance(
    seed=42,
    dataset=training_records,
    packages=("torch", "transformers", "peft"),
)

entry = project.capture(
    globals(),
    name="reproducible-run",
    parameters={"seed": 42},
    metadata=provenance.as_metadata(),
)
```

Provenance can include Python/platform information, best-effort Git commit and dirty state,
seed, selected package versions, and a deterministic SHA-256 fingerprint for a file or
JSON-like dataset value.

## Add resource and cost evidence

```python
from trainlens import attach_resource_profile, profile_resources

profile = profile_resources(
    duration_seconds=912.4,
    tokens_processed=3_200_000,
    gpu_memory_mb=18_450,
    estimated_cost_usd=1.27,
)

run_with_resources = attach_resource_profile(entry.run, profile)
```

Duration, throughput, token throughput, memory, and estimated cost become ordinary run metrics,
so they can be compared or used in Pareto objectives and constraints.

## Gate a candidate in CI

```bash
trainlens compare baseline.json candidate.json
trainlens check candidate.json --against baseline.json --require 'f1>=0.85'
```

`trainlens check` returns a non-zero exit code for material regressions, failed gates, or
missing required metrics.

## Extend deterministic analysis

Custom analyzers can be registered process-locally and selected with
`AnalysisConfig(analyzer="...")`. See [Analyzer plugins](analyzer-plugins.md).

For the optional LLM layer, see [LLM setup and privacy](llm-and-privacy.md) and
[Verified LLM improvement plans](verified-llm-plans.md).
