"""Generate dark-mode visual explanation examples for TrainLens."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trainlens.pipeline import explain_namespace  # noqa: E402
from trainlens.renderers.visual import DarkVisualRenderer  # noqa: E402


class VisionLanguageProjectorRun:
    vision_tower = object()
    language_model = object()
    mm_projector = object()

    class Config:
        model_type = "llava"

    config = Config()


def main() -> None:
    model = VisionLanguageProjectorRun()
    history = {
        "train_loss": [2.42, 2.04, 1.82, 1.74, 1.70],
        "eval_loss": [2.35, 2.12, 2.02, 2.01, 2.02],
        "eval_perplexity": [10.4, 8.6, 7.9, 7.8, 7.8],
    }
    training_trace = [
        {"step": 100, "epoch": 0.2, "event": "batch_end", "loss": 2.42, "lr": 0.00003},
        {"step": 200, "epoch": 0.4, "event": "batch_end", "loss": 2.04, "lr": 0.000028},
        {"step": 300, "epoch": 0.6, "event": "eval", "eval_loss": 2.12, "eval_perplexity": 8.6},
        {"step": 400, "epoch": 0.8, "event": "batch_end", "loss": 1.82, "lr": 0.000024},
        {"step": 500, "epoch": 1.0, "event": "eval", "eval_loss": 2.02, "eval_perplexity": 7.9},
        {
            "step": 600,
            "epoch": 1.2,
            "event": "checkpoint",
            "eval_loss": 2.01,
            "eval_perplexity": 7.8,
        },
    ]
    lora_rank = 4
    trainable_params = 8_000_000
    total_params = 7_000_000_000
    feature_names = ["projector_lr", "lora_rank", "caption_length", "image_resolution"]

    namespace = locals()
    result = explain_namespace(namespace)
    assets = DarkVisualRenderer().write_dashboard_assets(result, ROOT / "docs" / "assets")
    for name, path in assets.items():
        print(f"{name}: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
