"""Heuristics for LLM, CLIP, ViT, projector, and VLM fine-tuning."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trainlens.models.analysis import Recommendation, Signal
from trainlens.models.metric import MetricSeries

ARCHITECTURE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "llm": (
        "causallm",
        "seq2seq",
        "llama",
        "mistral",
        "gemma",
        "gpt",
        "qwen",
        "bert",
        "t5",
        "language_model",
    ),
    "clip": ("clip", "siglip", "contrastive"),
    "vit": ("vit", "visiontransformer", "deit", "swin"),
    "projector": ("projector", "mm_projector", "vision_projector", "resampler"),
    "vlm": ("vlm", "llava", "blip", "flamingo", "paligemma", "vision2seq", "vision_tower"),
}


def detect_foundation_architecture(model: object | None, namespace: Mapping[str, Any]) -> list[str]:
    """Infer foundation-model families from model attributes and notebook names."""

    haystack: list[str] = []
    if model is not None:
        haystack.extend(
            str(value).lower()
            for value in (
                model.__class__.__name__,
                getattr(model.__class__, "__module__", ""),
                getattr(getattr(model, "config", None), "model_type", ""),
                getattr(getattr(model, "config", None), "architectures", ""),
            )
        )
        for attr in (
            "vision_tower",
            "vision_model",
            "text_model",
            "language_model",
            "mm_projector",
        ):
            if hasattr(model, attr):
                haystack.append(attr)
    haystack.extend(name.lower() for name in namespace)
    text = " ".join(haystack)
    return [
        family
        for family, keywords in ARCHITECTURE_KEYWORDS.items()
        if any(word in text for word in keywords)
    ]


def detect_loss_plateau(loss: MetricSeries | None) -> Signal | None:
    if loss is None or loss.recent_slope is None or len(loss.values) < 4:
        return None
    if abs(loss.recent_slope) <= 0.01:
        return Signal(
            title="Loss plateau detected",
            detail=f"{loss.name} has barely moved over the last checkpoints.",
            severity="warning",
            evidence=(f"recent_slope={loss.recent_slope:.4f}",),
        )
    return None


def detect_contrastive_misalignment(series: Mapping[str, MetricSeries]) -> Signal | None:
    retrieval = _first_series(
        series,
        "validation_retrieval_recall",
        "retrieval_recall",
        "validation_recall_at_1",
    )
    loss = _first_series(
        series,
        "validation_contrastive_loss",
        "contrastive_loss",
        "validation_loss",
    )
    if retrieval and retrieval.last is not None and retrieval.last < 0.2:
        return Signal(
            title="Low contrastive retrieval quality",
            detail="Retrieval recall is low for a CLIP-style objective.",
            severity="warning",
            evidence=(f"{retrieval.name}={retrieval.last:.3f}",),
        )
    if loss and loss.delta is not None and loss.delta > 0:
        return Signal(
            title="Contrastive loss is regressing",
            detail="The image-text objective is getting worse instead of improving.",
            severity="critical",
            evidence=(f"delta={loss.delta:.3f}",),
        )
    return None


def detect_adapter_pressure(namespace: Mapping[str, Any]) -> Signal | None:
    rank = _numeric_value(namespace, "lora_r", "lora_rank", "adapter_rank")
    trainable = _numeric_value(namespace, "trainable_params", "trainable_parameters")
    total = _numeric_value(namespace, "total_params", "total_parameters")
    if rank is not None and rank <= 4:
        return Signal(
            title="Very small adapter rank",
            detail="LoRA/adapters may be under-capacity for the target task.",
            severity="warning",
            evidence=(f"rank={rank:g}",),
        )
    if trainable is not None and total is not None and total > 0:
        ratio = trainable / total
        if ratio < 0.001:
            return Signal(
                title="Tiny trainable parameter ratio",
                detail="The fine-tune updates less than 0.1% of parameters.",
                severity="warning",
                evidence=(f"trainable_ratio={ratio:.4%}",),
            )
    return None


def foundation_recommendations(families: list[str], signals: list[Signal]) -> list[Recommendation]:
    titles = {signal.title for signal in signals}
    recommendations: list[Recommendation] = []
    if "llm" in families:
        recommendations.append(
            Recommendation(
                action=(
                    "Track eval_loss, perplexity, learning-rate schedule, and "
                    "gradient norm per checkpoint."
                ),
                rationale=(
                    "LLM fine-tunes often look healthy in training loss while "
                    "drifting on held-out prompts."
                ),
                confidence=0.76,
            )
        )
    if "clip" in families:
        recommendations.append(
            Recommendation(
                action=(
                    "Evaluate image-to-text and text-to-image recall@k on a "
                    "frozen validation set."
                ),
                rationale=(
                    "CLIP fine-tunes need bidirectional retrieval checks, not "
                    "only loss curves."
                ),
                confidence=0.79,
            )
        )
    if "vit" in families:
        recommendations.append(
            Recommendation(
                action=(
                    "Check augmentation strength, layer-wise LR decay, and "
                    "validation accuracy by class."
                ),
                rationale="ViT fine-tunes are sensitive to data scale and augmentation mismatch.",
                confidence=0.7,
            )
        )
    if "projector" in families or "vlm" in families:
        recommendations.append(
            Recommendation(
                action=(
                    "Validate projector alignment with frozen vision and language "
                    "towers before full VLM tuning."
                ),
                rationale=(
                    "Projector instability can hide as noisy captioning or "
                    "instruction-following failures."
                ),
                confidence=0.82,
            )
        )
    if "Loss plateau detected" in titles:
        recommendations.append(
            Recommendation(
                action=(
                    "Inspect learning rate, warmup ratio, effective batch size, "
                    "and frozen module choices."
                ),
                rationale=(
                    "A plateau usually means the optimizer is no longer moving "
                    "useful parameters."
                ),
                confidence=0.73,
            )
        )
    return recommendations


def _first_series(series: Mapping[str, MetricSeries], *names: str) -> MetricSeries | None:
    for name in names:
        if name in series:
            return series[name]
    return None


def _numeric_value(namespace: Mapping[str, Any], *names: str) -> float | None:
    for name in names:
        value = namespace.get(name)
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
    return None
