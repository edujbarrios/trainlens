"""Minimal TrainLens usage outside a notebook for a VLM fine-tune."""

from trainlens.pipeline import explain_namespace
from trainlens.renderers.markdown import MarkdownRenderer


class LlavaLikeModel:
    vision_tower = object()
    language_model = object()
    mm_projector = object()

    class Config:
        model_type = "llava"

    config = Config()


namespace = {
    "model": LlavaLikeModel(),
    "history": {"train_loss": [2.4, 1.8, 1.62, 1.61], "eval_loss": [2.2, 1.9, 1.88, 1.88]},
    "lora_rank": 4,
    "trainable_params": 8_000_000,
    "total_params": 7_000_000_000,
}

print(MarkdownRenderer().render(explain_namespace(namespace)))
