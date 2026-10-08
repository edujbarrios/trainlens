# Python API

Install with `python -m pip install trainlens`. The default wheel includes
Matplotlib for plots, ReportLab for PDF, and the built-in OpenAI-compatible
request client. No TrainLens extras are needed. External ML frameworks and LLM
endpoints are not included.


## Explicit analysis selection

```python
from trainlens import AnalysisConfig, analyze

result = analyze(
    globals(),
    config=AnalysisConfig(
        model="model_v2",
        trainer="trainer_v2",
        strict=True,
        metrics={"validation_loss": validation_loss},
    ),
)
```

`AnalysisConfig` can select a model or trainer by notebook variable name or
concrete object. `metrics` and `labels` provide authoritative evidence when
name-based inference is not appropriate. With `strict=True`, ambiguous automatic
model selection raises instead of silently choosing one candidate.

## LLM reports

```python
from trainlens import ContextPolicy, build_improvement_ideas, build_paper_report

policy = ContextPolicy(
    max_metric_points=12,
    max_metric_series=20,
    max_variables=20,
    max_chars=32_000,
)
paper = build_paper_report(globals(), context_policy=policy)
ideas = build_improvement_ideas(globals(), context_policy=policy)
print(paper.markdown)
```

`build_llm_report` is an alias-style entry point for a paper report. A
`LiveReport` contains the generated Markdown and the extracted `AnalysisResult`.
The legacy `max_metric_points` argument remains supported. `ContextPolicy` adds
global budgets for metric series, notebook variables, training artifacts, model
candidates, and total rendered context. Omitted or truncated evidence is marked
explicitly.

Optional LLM reports include the deterministic TrainLens summary, signals,
evidence provenance, and recommendations. The provider explains those local
findings rather than replacing them with a separate interpretation of the
notebook.

### Inject an LLM provider

```python
from trainlens import LLMConfig, OpenAICompatibleProvider, build_paper_report

provider = OpenAICompatibleProvider(
    LLMConfig(
        base_url="http://localhost:11434/v1",
        model="llama3.2",
    )
)
report = build_paper_report(globals(), provider=provider)
```

`LLMProvider` is a public protocol, so custom transports can be injected without
changing TrainLens. `LLMConfig.api_key` is optional; the OpenAI-compatible
provider omits `Authorization` when no key is configured.

## Long-run plotting

```python
from trainlens import plot_training_curves

figure = plot_training_curves(
    {"log_history": trainer.state.log_history},
    metrics=("loss", "accuracy"),
    x_axis="auto",
    max_plot_points=500,
    show_best=True,
)
```

Plot thinning is deterministic, preserves bucket minima/maxima and endpoints,
and affects only the displayed points—not `training_curves()` or portable
run histories. `auto` selects recorded steps only when complete across
the plotted non-scalar series, otherwise it selects observation positions
for all plots. `step` mode retains gaps for unknown step coordinates.
Validation-best markers use known metric direction and never tune against
held-out test evidence.

## Inspect training configuration

```python
from trainlens import inspect_training_profile

profile = inspect_training_profile(model, trainer=trainer, namespace=globals())
print(profile.strategy)
print(profile.trainable_parameters, profile.total_parameters)
print(profile.trainable_fraction)
print(profile.adapters, profile.active_adapters)
print(profile.quantization)
```

`inspect_training_profile()` uses safe duck typing and does not import optional
framework packages. It normalizes common Hugging Face training arguments,
PEFT/LoRA/DoRA settings, and VLM configuration such as projector type,
multimodal learning rates, visual feature selection, and sequence length.

TrainLens distinguishes full and partial fine-tuning when parameter state is
available, reports trainable/total parameter counts and fractions, tracks
multiple and active PEFT adapters, and recognizes common 4-bit/8-bit
quantization flags. VLM profiles continue to report trainable and frozen vision,
language, and multimodal-projector components.

The normal `%explain_training` and report workflows use the same profile. The
exact sanitized profile included in an optional LLM request is visible through
`preview_notebook_context()` or `%explain_training --dry-run`.

## Register custom metric semantics

```python
from trainlens import MetricSpec, register_metric

register_metric(
    MetricSpec(
        name="dice",
        direction="higher",
        lower_bound=0.0,
        upper_bound=1.0,
        aliases=("dice_score",),
    )
)
```

The process-wide `MetricRegistry` is used by comparison and experiment-planning
helpers. A metric spec can define direction, bounds, aliases, units, and custom
material-change thresholds. Built-in metric behavior remains available through
`default_metric_registry()`.

## Portable runs and richer comparisons

```python
from trainlens import compare_runs, load_run, save_run, training_run_from_analysis

run = training_run_from_analysis(result, parameters={"learning_rate": 1e-4})
save_run(run, "run-a.json")
restored = load_run("run-a.json")
comparison = compare_runs(run, restored)
```

`TrainingRun` JSON preserves metric histories and step alignment together with
portable parameters/metadata. Comparing two `TrainingRun` objects reports final
metric changes, changed parameters, best trajectory values, and observation
counts. Mapping and `AnalysisResult` comparisons remain supported.

## Plan controlled and multi-objective experiments

```python
from trainlens import (
    ExperimentRun,
    MetricConstraint,
    ObjectiveSpec,
    experiment_config,
    pareto_front,
    suggest_multiobjective_experiment,
    suggest_next_experiment,
)

runs = [ExperimentRun(
    name="baseline",
    metrics={"train_loss": 0.20, "validation_loss": 0.50, "latency_ms": 9.0},
    parameters={"learning_rate": 1e-3, "dropout": 0.10, "batch_size": 32},
    estimated_cost="medium",
)]

single = suggest_next_experiment(
    runs, objective_metric="validation_loss", minimum_improvement=0.01
)
next_parameters = experiment_config(single, base_parameters=runs[0].parameters)

objectives = (
    ObjectiveSpec(metric="validation_loss", direction="min"),
    ObjectiveSpec(metric="latency_ms", direction="min"),
)
constraints = (
    MetricConstraint(metric="latency_ms", operator="<=", threshold=10.0),
)
front = pareto_front(runs, objectives, constraints=constraints)
multi = suggest_multiobjective_experiment(
    runs,
    objectives,
    constraints=constraints,
)
```

Single-objective recommendations remain deterministic controlled heuristics.
The multi-objective API can filter hard constraints and reason over a Pareto
front instead of forcing all metrics into one scalar score.

## Extensible live monitoring

```python
from trainlens import TrainingAlert, TrainLensMonitor


def gradient_detector(history, current, config):
    value = current.metrics.get("gradient_norm")
    if value is None or value <= 10:
        return ()
    return (
        TrainingAlert(
            code="high_gradient_norm",
            severity="warning",
            message="Gradient norm is high.",
            evidence=(f"gradient_norm={value}",),
            step=current.step,
            persistent=True,
        ),
    )


monitor = TrainLensMonitor(detectors=(gradient_detector,))
```

Custom `AlertDetector` callables receive immutable history, the current
observation, and `MonitorConfig`. Persistent custom alerts use the same
re-arm-after-recovery lifecycle as built-in stagnation/overfitting alerts.

## Public entry points

The stable top-level package exports explicit analysis configuration, report and
provider controls, metric semantics, portable runs/comparison helpers,
experiment planning, monitoring/callback classes, `TrainingProfile`,
`inspect_training_profile()`, and export helpers. Framework inspection internals
and model dataclasses under subpackages should be treated as lower-level APIs.
