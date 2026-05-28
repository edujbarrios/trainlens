# Notebooks

These notebooks show how to use TrainLens from a cloned repository without
installing a package into the active environment.

## Examples

- `llm_finetune_report.ipynb`: LoRA/LLM fine-tuning diagnostics.
- `vlm_projector_report.ipynb`: VLM projector alignment and loss plateau checks.
- `clip_contrastive_report.ipynb`: CLIP-style contrastive loss and retrieval checks.

Each notebook uses:

```python
from pathlib import Path
import sys

TRAINLENS_REPO = Path("..").resolve()
sys.path.insert(0, str(TRAINLENS_REPO / "src"))
```

That makes the local source tree importable inside the notebook without running
`pip install`.
