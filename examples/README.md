# TrainLens Example

This folder contains one notebook:

- `cpu_training_with_trainlens.ipynb`: trains a tiny CPU-only classifier, keeps
  the training metrics in notebook memory, and asks TrainLens to explain the
  result with an OpenAI-compatible LLM provider.

Before running the final TrainLens cell, set your LLM7 API key when prompted.
The notebook configures:

```python
TRAINLENS_LLM_BASE_URL = "https://api.llm7.io/v1"
TRAINLENS_LLM_MODEL = "gpt-5.4"
```

The key is read with `getpass` and is not stored in the notebook.
