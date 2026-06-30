"""PyTorch-style training loop metrics without requiring PyTorch as a dependency."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trainlens.pipeline import explain_namespace  # noqa: E402
from trainlens.renderers.markdown import MarkdownRenderer  # noqa: E402


class ScalarTensor:
    """Small tensor-like scalar used to mimic torch.Tensor.item()."""

    def __init__(self, value: float) -> None:
        self.value = value

    def item(self) -> float:
        return self.value


class TorchLikeClassifier:
    def parameters(self) -> list[object]:
        return [object()]

    def state_dict(self) -> dict[str, float]:
        return {"encoder.weight": 0.42}


TorchLikeClassifier.__module__ = "torch.nn.modules.module"


model = TorchLikeClassifier()
train_losses = [2.3, 1.9, 1.55, 1.34]
val_losses = [2.4, 2.0, 1.78, 1.72]
epoch_logs = [
    {"epoch": 1, "train_loss": 2.3, "val_loss": 2.4},
    {"epoch": 2, "train_loss": 1.9, "val_loss": 2.0},
    {"epoch": 3, "train_loss": 1.55, "val_loss": 1.78},
    {"epoch": 4, "train_loss": 1.34, "val_loss": 1.72},
]
callback_metrics = {
    "train_loss": ScalarTensor(1.34),
    "val_loss": ScalarTensor(1.72),
}

result = explain_namespace(globals())
print(MarkdownRenderer().render(result))
