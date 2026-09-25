# Notebook workflow and framework adapters

TrainLens scans a namespace for metric histories, model-like objects, labels,
and lightweight metadata. Common metric aliases such as `eval_loss`,
`val_loss`, `validation_loss`, `train/accuracy`, and `eval-accuracy` are
normalized.

## Supported evidence

- Dictionaries, sequences, log histories, and trace-like metric events
- Keras history objects
- Hugging Face `trainer.state.log_history`
- PyTorch Lightning callback and logged metrics
- Plain PyTorch-like model, optimizer, scheduler, loader, and dataset objects
- Labels named like `y_train`, `labels`, or `target`

Adapters use duck typing and module names, so TensorFlow, PyTorch,
Transformers, and Lightning are not required TrainLens dependencies.

## Magic commands

```python
%load_ext trainlens.magic.extension

%explain_training --name baseline --no-llm
%explain_training --name candidate
%compare_runs baseline candidate
%suggest_improvements
```

`%explain_training` always analyzes and captures the local run before it asks an
LLM provider for an explanation. A missing or failing provider therefore does
not discard the captured run. Use `--no-llm` to capture and render the local
analysis without any provider request.

Run names are stable notebook-local labels. `%compare_runs BASELINE EXPERIMENT`
accepts names or one-based capture indices. With no arguments, `%compare_runs`
keeps the original behavior and compares the two most recent runs.

## Preview exactly what leaves the kernel

Before making a provider request, preview the same sanitized context TrainLens
would send:

```python
%explain_training --dry-run
```

or from Python:

```python
from trainlens import preview_notebook_context

preview_notebook_context()
```

Preview mode uses the production context builder and sanitizer and performs no
network request. It does not require provider configuration. Metric histories
include aligned steps or fractional epochs when available, while long histories
remain bounded by the configured metric-point budget.

## Native notebook display

`LiveReport` and notebook context previews implement IPython's Markdown display
protocol. They can therefore be the final expression of a cell:

```python
from trainlens import build_paper_report

build_paper_report()
```

The object still exposes `.markdown` and `.result` for programmatic use.

## Framework examples

```python
# Keras
history = model.fit(x_train, y_train, validation_data=(x_val, y_val))

# Hugging Face
trainer.train()

# Lightning
trainer.fit(module, datamodule=datamodule)

# Keep plain PyTorch objects visible in the notebook namespace:
model, optimizer, scheduler, train_loader

%explain_training --name first-run
```

Keep descriptive variables small and clearly named. TrainLens deliberately
avoids serializing arbitrary objects or full datasets. If the same model is
detected through generic and framework-specific introspection, TrainLens merges
that evidence into one model candidate rather than duplicating prompt context.

## Explicit namespaces

Outside an active IPython shell, pass a mapping explicitly:

```python
from trainlens import build_paper_report, preview_notebook_context

namespace = {"history": history, "model_name": "classifier-v2"}
preview = preview_notebook_context(namespace)
report = build_paper_report(namespace)
```

Calling these helpers without a namespace outside IPython raises an error
because there is no notebook namespace to inspect.

## Extension reload behavior

Loading the extension repeatedly is idempotent. Unloading removes the three
line magics; loading it again in the same IPython shell reuses the TrainLens
magic instance, so notebook-local captured runs remain available across an
extension reload.
