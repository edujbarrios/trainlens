# TrainLens

**TrainLens turns training runs into evidence: what changed, what went wrong, and what experiment to run next.**

TrainLens is a lightweight, notebook-first toolkit for understanding ML training runs and
small experiment histories. It keeps train, validation, and held-out test evidence distinct,
compares runs with metric-aware semantics, explains aggregate dataset context, and helps turn
observations into controlled next experiments. Core analysis is local and deterministic; LLM
support is optional.

<p align="center">
  <a href="https://github.com/edujbarrios/trainlens/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/edujbarrios/trainlens/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://pypi.org/project/trainlens/0.15.0/"><img alt="PyPI 0.15.0" src="https://img.shields.io/badge/pypi-0.15.0-blue?logo=pypi"></a>
  <a href="pyproject.toml"><img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-blue"></a>
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-yellow"></a>
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/edujbarrios/trainlens/blob/main/examples/quickstart.ipynb">
    <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
  </a>
</p>

The Colab is the fastest way to see the complete workflow: **training curves → deterministic
diagnosis → dataset and fine-tuning context → optional sanitized LLM review → concrete next
experiments**.

## See the full workflow in a few cells

Install TrainLens in a normal notebook. No tracking server or hosted service is required.

```python
%pip install -q "trainlens[plots]"
```

Start from the objects you already have after training. TrainLens understands common metric
aliases and keeps training, validation, and held-out test evidence in their proper roles.

```python
history = {
    "train_loss":     [1.31, 0.93, 0.66, 0.47, 0.34, 0.25],
    "val_loss":       [1.28, 0.91, 0.70, 0.58, 0.55, 0.57],
    "train_accuracy": [0.55, 0.68, 0.79, 0.87, 0.92, 0.95],
    "val_accuracy":   [0.54, 0.66, 0.75, 0.80, 0.82, 0.81],
}

test_metrics = {
    "loss": 0.64,
    "accuracy": 0.75,
    "f1": 0.74,
    "precision": 0.77,
    "recall": 0.72,
}
```

Analyze the notebook state locally and plot the learning curves:

```python
from trainlens import analyze, plot_training_curves, render_report

analysis = analyze(globals())
plot_training_curves(globals())
print(render_report(analysis))
```

From this small example, TrainLens has enough evidence to reason about more than a final
score. It can preserve the final test metrics, identify a widening train/validation gap,
notice late validation degradation, compare validation behavior with held-out test behavior,
and keep the test split out of tuning decisions. If there is no test result yet, it can instead
surface that final held-out evaluation is still missing.

On real fine-tuning notebooks, the same analysis can also inspect supported model and trainer
objects for details such as trainable versus frozen parameters, PEFT adapters, LoRA rank,
alpha and target modules, quantization, learning rates, and multimodal/VLM settings.

## Explain the dataset behind the metrics

Model results are easier to interpret when the dataset is part of the evidence. `explain_dataset()`
creates a deterministic aggregate profile without requiring pandas, Hugging Face Datasets, Pillow,
NumPy, PyTorch, or an external service:

```python
from trainlens import explain_dataset

dataset_context = explain_dataset(
    {
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset,
    },
    target="label",
)

dataset_context
```

Besides tabular and text data, TrainLens understands image-centered datasets and multimodal
image+text / image+tabular inputs. It can inspect PIL-like images, image-shaped arrays or tensors,
Hugging Face-style image values, image paths, and multiple images per row through metadata only.

```python
vlm_context = explain_dataset(
    {
        "image": images,
        "caption": captions,
        "label": labels,
    },
    target="label",
    name="train",
)

print(vlm_context.is_multimodal)
print(vlm_context.modality_summaries[0].modalities)  # ('image', 'text')
```

For image features, TrainLens can summarize image counts, available width/height ranges, aspect
ratios, channel counts, modes, formats, images per row, variable dimensions, and material
resolution differences across splits. It never renders pixels, image bytes, private paths, or
filenames into the explanation.

The same profile still includes row counts, missingness, cardinality, target distributions,
numeric/text feature statistics, schema differences, and cross-split observations. All of this
is descriptive evidence: TrainLens can flag a resolution or class-distribution difference, but it
does **not** claim that the dataset property caused the observed model behavior.

You can use the profile on its own or explicitly attach it to an LLM explanation. TrainLens
never silently adds the raw dataset or underlying images to an outbound request.

## Turn the diagnosis into an experiment decision

A useful training analysis should answer **what should I try next?**, not just describe a
curve. TrainLens can preserve runs, select checkpoints from validation evidence, compare
experiments, and build a small local experiment history.

```python
from trainlens import Project, select_checkpoint

project = Project(".trainlens")
entry = project.capture(
    globals(),
    name="lr-2e-5-seed-1",
    parameters={
        "learning_rate": 2e-5,
        "seed": 1,
        "lora_r": 16,
    },
)

selection = select_checkpoint(entry.run, metric="validation_loss")
print(selection.best_step)
print(selection.best_value)
print(selection.suggested_patience)
```

As runs accumulate, TrainLens can rank them with objectives and constraints, mark
Pareto-efficient configurations, aggregate repeated seeds, compare observed improvements with
between-run variability, and estimate parameter effects only from controlled pairs that differ
in one recorded parameter.

That means a notebook can evolve from:

```text
"run finished; val_loss=0.55"
```

into evidence such as:

```text
- validation loss improved until the late stage, then degraded
- training kept improving while validation stopped improving
- held-out test performance is weaker than the best validation behavior
- the target is imbalanced and validation has a different observed class mix
- checkpoint selection should use validation evidence, not the test split
- the next run should change one controlled parameter and define a measurable success criterion
```

## Optional: ask an LLM without hiding the evidence

The LLM layer is downstream of TrainLens' deterministic analysis. You choose the provider, and
you can inspect the exact sanitized evidence before any request is sent.

```python
from getpass import getpass

from trainlens import OpenAICompatibleProvider

llm = OpenAICompatibleProvider.from_values(
    base_url="https://your-provider.example/v1",
    model="your-model",
    api_key=getpass("LLM API key: "),
)
```

Define what kind of analysis you want in normal prose:

```python
from trainlens import improvement_plan_prompt, prompt_text

prompt = improvement_plan_prompt(
    objective=prompt_text("""
        Explain what happened during this fine-tuning run and propose the three
        highest-value next experiments.

        Use the test split only as final generalization evidence; do not tune to it.
        Use dataset properties as context, not as proof of causation.
        Prefer controlled, low-cost changes first.
    """),
    model_family="LLM fine-tuning with PEFT/LoRA",
    focus_areas=(
        "train/validation generalization gap",
        "validation-to-test degradation",
        "dataset imbalance or split differences",
        "checkpoint selection and early stopping",
        "learning-rate schedule",
        "regularization and adapter capacity",
    ),
)
```

Preview the request first. This step performs no network call:

```python
from trainlens import preview_llm_request

preview = preview_llm_request(
    globals(),
    mode="improvement_ideas",
    provider=llm,
    prompt_options=prompt,
    dataset_explanation=dataset_context,
)

print(preview.system_prompt)
print(preview.user_prompt)  # sanitized notebook + aggregate dataset evidence
```

Then generate an evidence-backed explanation and experiment plan using the same context:

```python
from trainlens import build_improvement_ideas

report = build_improvement_ideas(
    globals(),
    provider=llm,
    prompt_options=prompt,
    dataset_explanation=dataset_context,
)

report
```

For machine-checkable workflows, TrainLens also supports structured plans whose model and
dataset evidence IDs are verified against deterministic TrainLens evidence rather than accepting
invented citations:

```python
from trainlens import build_verified_improvement_plan

plan = build_verified_improvement_plan(
    globals(),
    provider=llm,
    dataset_explanation=dataset_context,
)
print(plan.recommendations)
print(plan.unsupported_evidence_ids)
```

Open the full notebook to run this end to end:
[**TrainLens quickstart in Google Colab**](https://colab.research.google.com/github/edujbarrios/trainlens/blob/main/examples/quickstart.ipynb).

## What TrainLens can do

- **Analyze train / validation / test evidence** without collapsing their roles.
- **Explain tabular, image, and multimodal datasets locally** with aggregate split, schema,
  missingness, target-balance, image dimensions/modalities, and cross-split evidence without
  exposing raw rows, pixels, image bytes, or private file paths.
- **Inspect fine-tuning configuration** across common Hugging Face, PEFT/LoRA, Keras,
  Lightning, PyTorch-style, and VLM workflows through dependency-light introspection.
- **Plot and monitor training** with learning curves, alerts, and framework callbacks.
- **Select checkpoints deterministically** from training/validation trajectories while
  rejecting held-out test metrics for tuning.
- **Compare experiments** pairwise or through multi-run leaderboards with objectives,
  constraints, and Pareto fronts.
- **Reason about repeated runs** with mean/spread summaries and conservative signal-to-noise
  comparisons rather than overclaiming statistical significance.
- **Estimate parameter effects** from controlled one-parameter run pairs.
- **Capture reproducibility evidence** such as Git revision, seed, environment, package
  versions, and dataset fingerprints.
- **Track resource and cost metrics** such as duration, throughput, memory, and estimated
  cost so they can participate in experiment objectives.
- **Store runs locally** with `Project`, or save/load portable run JSON directly.
- **Gate regressions in CI** with the `trainlens` command-line interface.
- **Extend analysis** with process-local analyzer plugins.
- **Use optional LLMs** with request previewing, privacy controls, reusable prompt recipes,
  aggregate dataset context, and evidence-verified structured plans.
- **Export reports** to Markdown, JSON, HTML, and optional PDF.

## Compare or gate runs from the CLI

Portable runs are useful outside notebooks too:

```bash
trainlens compare baseline.json candidate.json
trainlens check candidate.json --against baseline.json --require 'f1>=0.85'
```

`trainlens check` exits non-zero on material regressions, failed metric gates, or missing
required metrics, so TrainLens evidence can participate directly in CI.

## Documentation

- [Getting started](docs/en/getting-started.md)
- [Dataset-aware analysis](docs/en/dataset-context.md)
- [Experiment workflows](docs/en/experiment-workflows.md)
- [Notebook workflows](docs/en/notebooks.md)
- [Python API](docs/en/python-api.md)
- [Monitoring](docs/en/monitoring.md)
- [Analyzer plugins](docs/en/analyzer-plugins.md)
- [LLM setup and privacy](docs/en/llm-and-privacy.md)
- [Verified LLM plans](docs/en/verified-llm-plans.md)
- [Exports and troubleshooting](docs/en/exports-and-troubleshooting.md)
- [Documentación en español](docs/es/README.md)

## Privacy and scope

Local model and dataset analysis does not make an LLM request. Dataset explanations are aggregate
profiles and do not contain raw feature rows, pixels, image bytes, or private image paths. When
LLM support is enabled, TrainLens bounds and redacts notebook evidence, and dataset context is
only included when a `DatasetExplanation` is explicitly supplied. Generated explanations should
still be verified against the underlying data.

TrainLens is an experiment-understanding layer, not a full MLOps platform, tracker, profiler,
or causal diagnosis system. It is designed for research and small-to-medium workflows where
the important training context already lives in Python and Jupyter.

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md) to contribute and [SECURITY.md](SECURITY.md) to report
vulnerabilities. TrainLens is licensed under the [Apache License 2.0](LICENSE).
