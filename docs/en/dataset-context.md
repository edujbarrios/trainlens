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
with `columns`, Hugging Face-like objects with `column_names`, and mappings of named splits.
TrainLens does not require pandas, Hugging Face Datasets, Pillow, NumPy, or PyTorch for dataset
explanations. Optional objects from those libraries are inspected through small public metadata
surfaces when they are already present in your notebook.

For ordinary tabular/text features, TrainLens can summarize:

- row count;
- detected numeric, categorical, text, boolean, mixed, or object-like features;
- missing values and cardinality;
- numeric minimum, maximum, and mean;
- average text length;
- supervised target distribution or continuous-target range;
- constant features and material missingness observations.

Across splits it can point out feature-schema differences and material changes in the observed
target distribution.

## Image datasets

`explain_dataset()` also recognizes image-centered datasets without decoding or transmitting the
media. It can inspect common representations such as:

- PIL-like image objects exposing `size`, `mode`, and optionally `format`;
- NumPy/PyTorch-like arrays or tensors exposing image-shaped `shape` metadata (`H×W`, `H×W×C`,
  or `C×H×W`);
- common image-path strings such as `.jpg`, `.png`, `.webp`, `.tiff`, or `.heic`;
- Hugging Face-style image values such as `{"path": ..., "bytes": ...}`;
- a sequence of multiple images stored in one dataset row.

For recognized image features, the returned `MultimodalDatasetExplanation` exposes aggregate
`DatasetImageSummary` evidence including image counts, images per row, available width/height
ranges and means, aspect ratios, channel counts, image modes, and file formats.

```python
from trainlens import explain_dataset

image_context = explain_dataset(
    {
        "image": images,
        "label": labels,
    },
    target="label",
    name="train",
)

print(image_context.modality_summaries[0].image_features[0])
```

TrainLens can also flag variable dimensions, multiple images per row, mixed observed formats,
and material mean image-size differences across splits. These observations are descriptive: a
resolution difference between train and validation may matter, but it is not automatically the
cause of a metric gap.

## Multimodal datasets

When a split combines image features with another input modality, TrainLens marks it explicitly
as image-centered multimodal data. Current modality summaries distinguish:

- `image`;
- `text`;
- `tabular`.

For example, an image-caption dataset can be explained directly:

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

print(vlm_context.is_multimodal)                # True
print(vlm_context.modality_summaries[0].modalities)  # ('image', 'text')
```

The same API handles image + metadata datasets and rows that contain several images, which is
useful for VLM, visual question answering, image comparison, document-image, and other
image-centered training workflows.

## Use dataset context to explain model results

The returned `DatasetExplanation` (including `MultimodalDatasetExplanation`) can be passed
explicitly to notebook/LLM helpers:

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

Dataset facts receive stable evidence IDs alongside metric and training-signal evidence. Image
features are represented as aggregate feature evidence (`kind=image`), while observations such
as variable dimensions, image+text modality combinations, and cross-split resolution changes
also receive deterministic dataset evidence IDs. This lets structured recommendations cite
visual context without accepting invented evidence.

## Privacy model

`explain_dataset()` is local and deterministic. Its rendered explanation contains aggregate
statistics and observations, not raw feature rows. For image features specifically, TrainLens
does **not** render pixel values, image bytes, private paths, or filenames. A path may be used
locally to recognize an image extension, but the path itself is not included in the explanation.

TrainLens does not automatically inspect a dataset and send it to an LLM. Dataset context is
included in an outbound request only when you explicitly pass a `DatasetExplanation` to a report
or preview helper. The outbound context contains the aggregate explanation, not the underlying
images.

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
- train and test expose different schemas;
- an image feature has variable dimensions or multiple images per row;
- train and validation use materially different mean image sizes;
- a split combines image and text/tabular inputs.

Those facts can make a model result easier to reason about, but they do **not** establish that a
dataset property caused a training failure or performance gap. Treat dataset observations as
hypotheses and experiment context, not causal conclusions.
