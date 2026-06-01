# Live Notebook Cells

These cells show the shortest path from training-loop variables to a live TrainLens report.

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

## 3. Render Markdown And Visual Dashboard

```python
from IPython.display import HTML, Markdown, display

from trainlens.pipeline import explain_namespace
from trainlens.renderers.markdown import MarkdownRenderer
from trainlens.renderers.visual import DarkVisualRenderer


result = explain_namespace(globals())
display(Markdown(MarkdownRenderer().render(result)))
display(HTML(DarkVisualRenderer().render_dashboard_html(result)))
```

## 4. Use Magic Commands

```python
%load_ext trainlens.magic.extension
%explain_training
%training_dashboard
%compare_runs
```

The magic commands read the active notebook namespace, so they work after your
training cell has created `model`, `history`, `train_losses`, `val_losses`,
`epoch_logs`, `training_trace`, or similar variables.
