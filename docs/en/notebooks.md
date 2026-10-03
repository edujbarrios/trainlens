# Notebook workflow and framework adapters

TrainLens scans a namespace for metric histories, model-like objects, labels,
and lightweight metadata. Common metric aliases such as `eval_loss`,
`val_loss`, `validation_loss`, `train/accuracy`, `eval-accuracy`, `test_loss`,
and `test_f1` are normalized while preserving the train / validation / test
split.

## Supported evidence

- Dictionaries, sequences, log histories, and trace-like metric events
- Split metric containers such as `train_metrics`, `validation_metrics`, and `test_metrics`
- Keras history objects
- Hugging Face `trainer.state.log_history`
- PyTorch Lightning callback and logged metrics
- Plain PyTorch-like model, optimizer, scheduler, loader, and dataset objects
- Labels named like `y_train`, `labels`, or `target`

Adapters use duck typing and module names, so TensorFlow, PyTorch,
Transformers, and Lightning are not required TrainLens dependencies.

## Train / validation / test results

TrainLens keeps split roles explicit instead of flattening all results into one
mapping. For example:

```python
history = {
    "train_loss": [1.2, 0.7, 0.4],
    "val_loss": [1.1, 0.72, 0.55],
    "train_accuracy": [0.62, 0.80, 0.91],
    "val_accuracy": [0.60, 0.77, 0.84],
}

test_metrics = {
    "loss": 0.61,
    "accuracy": 0.81,
    "f1": 0.80,
    "precision": 0.82,
    "recall": 0.79,
}

from trainlens import analyze

result = analyze(globals())
result.metrics
```

The structured result promotes final metrics such as `validation_f1`,
`test_precision`, or `test_perplexity`, not only loss and accuracy. Direction-aware
train/validation gaps can identify possible overfitting even for loss-only runs.
A material validation-to-test degradation is reported separately because it can
indicate validation over-selection, split mismatch, or distribution shift. If
validation evidence exists but no test result is detected, TrainLens can recommend
a final held-out evaluation.

## Plot learning curves

Plotting is optional so the core package remains lightweight:

```python
%pip install -q "trainlens[plots]"
```

Then:

```python
from trainlens import plot_training_curves, training_curves

plot_training_curves(globals(), metrics=("loss", "accuracy"))
curves = training_curves(globals())  # structured plot-ready data
```

The helper plots train and validation histories and places a scalar held-out test
result at the final observation when appropriate. It returns the Matplotlib
`Figure`, so normal Matplotlib customization and saving still work.

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

Preview only the sanitized notebook evidence:

```python
%explain_training --dry-run
```

or:

```python
from trainlens import preview_notebook_context

preview_notebook_context()
```

When you also want to inspect the trusted system instructions and the exact
OpenAI-compatible request messages, use the configured provider path:

```python
from trainlens import preview_llm_request

preview = preview_llm_request(
    globals(),
    mode="improvement_ideas",
    provider=llm,
    prompt_options=prompt,
)
print(preview.system_prompt)
print(preview.user_prompt)
```

Both preview paths perform no network request. Metric histories include aligned
steps or fractional epochs when available, while long histories remain bounded
by the configured context policy.

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
test_metrics = dict(zip(model.metrics_names, model.evaluate(x_test, y_test)))

# Hugging Face
trainer.train()
eval_metrics = trainer.evaluate()
test_output = trainer.predict(test_dataset)
test_metrics = test_output.metrics

# Lightning
trainer.fit(module, datamodule=datamodule)
test_metrics = trainer.test(module, datamodule=datamodule)[0]

# Keep plain PyTorch objects and normal result dictionaries visible:
model, optimizer, scheduler, train_loader
history, test_metrics
```

With real model/trainer objects in the namespace, TrainLens can also inspect
supported fine-tuning configuration such as trainable/frozen parameters,
PEFT/LoRA adapters, rank/alpha/target modules, quantization, learning rates, and
common multimodal settings.

Keep descriptive variables small and clearly named. TrainLens deliberately
avoids serializing arbitrary objects or full datasets. If the same model is
detected through generic and framework-specific introspection, TrainLens merges
that evidence into one model candidate rather than duplicating prompt context.

## Explicit namespaces

Outside an active IPython shell, pass a mapping explicitly:

```python
from trainlens import build_paper_report, preview_notebook_context

namespace = {"history": history, "test_metrics": test_metrics}
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
