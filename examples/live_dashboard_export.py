"""Export a live-style TrainLens dashboard to HTML."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trainlens.pipeline import explain_namespace  # noqa: E402
from trainlens.renderers.markdown import MarkdownRenderer  # noqa: E402
from trainlens.renderers.visual import DarkVisualRenderer  # noqa: E402


class AdapterFineTune:
    active_adapters = ["vision_projector"]
    feature_importances_ = [0.18, 0.31, 0.26, 0.25]

    class Config:
        model_type = "mistral"

    config = Config()


model = AdapterFineTune()
history = {
    "train_loss": [2.8, 2.2, 1.94, 1.86],
    "eval_loss": [2.7, 2.3, 2.18, 2.17],
    "eval_perplexity": [14.9, 10.2, 8.9, 8.8],
}
training_trace = [
    {"step": 100, "event": "batch_end", "loss": 2.8, "lr": 0.00003},
    {"step": 200, "event": "batch_end", "loss": 2.2, "lr": 0.000028},
    {"step": 300, "event": "eval", "eval_loss": 2.18, "eval_perplexity": 8.9},
    {"step": 400, "event": "checkpoint", "message": "saved adapter weights"},
]
feature_names = ["learning_rate", "lora_rank", "caption_length", "batch_size"]
lora_rank = 4
trainable_params = 9_000_000
total_params = 7_000_000_000

result = explain_namespace(globals())
renderer = DarkVisualRenderer()
html = renderer.render_dashboard_html(result)
output_path = ROOT / "examples" / "generated" / "live-dashboard.html"
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(html, encoding="utf-8")

print(MarkdownRenderer().render(result))
print(f"Wrote {output_path.relative_to(ROOT)}")
