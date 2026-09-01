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
%explain_training
%suggest_improvements
%compare_runs
```

The first two commands require a configured LLM endpoint. `%compare_runs`
compares the two most recently captured runs; it reports that more runs are
needed if fewer than two are available. Command-line arguments are currently
ignored.

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

%explain_training
```

Keep descriptive variables small and clearly named. TrainLens deliberately
avoids serializing arbitrary objects or full datasets.

## Explicit namespaces

Outside an active IPython shell, pass a mapping explicitly:

```python
from trainlens import build_paper_report

report = build_paper_report({"history": history, "model_name": "classifier-v2"})
```

Calling `build_paper_report()` without a namespace outside IPython raises an
error because there is no notebook namespace to inspect.

