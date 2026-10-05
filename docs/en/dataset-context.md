# Dataset-aware analysis

Model metrics are easier to interpret when the dataset behind them is visible as evidence.
TrainLens provides `explain_dataset()` for a deterministic, dependency-light profile that can be
reviewed on its own or explicitly attached to model-result explanations.

## Explain a dataset locally

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

Supported inputs include mappings of columns, sequences of row mappings, pandas-like objects
with `columns`, Hugging Face-like objects with `column_names`, and mappings of named splits to
those tabular forms. TrainLens does not require pandas or Hugging Face Datasets to provide this
feature.

For each split, TrainLens can summarize:

- row count;
- detected numeric, categorical, text, boolean, mixed, or object-like features;
- missing values and cardinality;
- numeric minimum, maximum, and mean;
- average text length;
- supervised target distribution or continuous-target range;
- constant features and material missingness observations.

Across splits it can point out feature-schema differences and material changes in the observed
target distribution.

## Use dataset context to explain model results

The returned `DatasetExplanation` can be passed explicitly to notebook/LLM helpers:

```python
preview = preview_llm_request(
    globals(),
    mode="improvement_ideas",
    provider=llm,
    dataset_explanation=dataset_context,
)

print(preview.user_prompt)
```

Then reuse exactly the same aggregate context when generating a report:

```python
report = build_improvement_ideas(
    globals(),
    provider=llm,
    dataset_explanation=dataset_context,
)
```

Or use it in a structured, evidence-verified plan:

```python
plan = build_verified_improvement_plan(
    globals(),
    provider=llm,
    dataset_explanation=dataset_context,
)
```

Dataset facts receive stable evidence IDs alongside metric and training-signal evidence. This
lets structured recommendations cite facts such as split sizes, target balance, feature
statistics, and cross-split observations without accepting invented dataset citations.

## Privacy model

`explain_dataset()` is local and deterministic. Its rendered explanation contains aggregate
statistics and observations, not raw feature rows. TrainLens does not automatically inspect a
dataset and send it to an LLM. Dataset context is included in an outbound request only when you
explicitly pass a `DatasetExplanation` to a report or preview helper.

Always use `preview_llm_request()` when you want to inspect the exact request before it is sent.
The combined notebook and dataset context is also bounded by `ContextPolicy`.

Target labels may appear in class-distribution summaries because they are part of the requested
aggregate target evidence. If label names themselves are sensitive, transform or anonymize them
before creating the explanation.

## Interpretation limits

Dataset evidence is descriptive. For example, TrainLens may report that:

- the target is strongly imbalanced;
- validation contains a different observed class mix than training;
- a feature has substantial missingness;
- train and test expose different schemas.

Those facts can make a model result easier to reason about, but they do **not** establish that a
dataset property caused a training failure or performance gap. Treat dataset observations as
hypotheses and experiment context, not causal conclusions.
