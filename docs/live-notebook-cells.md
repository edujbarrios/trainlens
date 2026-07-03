# LLM Notebook Cells

These cells show the shortest path from training-loop variables to a TrainLens
LLM report.

## 1. Load TrainLens From A Clone

```python
from pathlib import Path
import sys

TRAINLENS_REPO = Path("/path/to/trainlens").resolve()
sys.path.insert(0, str(TRAINLENS_REPO / "src"))
```

## 2. Create Or Finish A Training Run

```python
model = my_model
train_losses = [2.3, 1.9, 1.55, 1.34]
val_losses = [2.4, 2.0, 1.78, 1.72]
epoch_logs = [
    {"epoch": 1, "train_loss": 2.3, "val_loss": 2.4},
    {"epoch": 2, "train_loss": 1.9, "val_loss": 2.0},
    {"epoch": 3, "train_loss": 1.55, "val_loss": 1.78},
    {"epoch": 4, "train_loss": 1.34, "val_loss": 1.72},
]
```

## 3. Configure The LLM Provider

```python
import os

os.environ["TRAINLENS_LLM_BASE_URL"] = "https://api.openai.com/v1"
os.environ["TRAINLENS_LLM_API_KEY"] = "your-api-key"
os.environ["TRAINLENS_LLM_MODEL"] = "gpt-4.1-mini"
```

## 4. Render The Notebook Report

```python
from trainlens.notebook import display_llm_report


report = display_llm_report(globals())
```

`report.result` keeps the extracted metric values, while `report.markdown`
keeps the LLM-generated notebook report.

## 5. Use Magic Commands

```python
%load_ext trainlens.magic.extension
%explain_training
%compare_runs
```

The helper and magic read the active notebook namespace, so they work after your
training cell has created `model`, `history`, `train_losses`, `val_losses`,
`epoch_logs`, `training_trace`, or similar variables.
