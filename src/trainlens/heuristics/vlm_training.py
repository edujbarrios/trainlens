"""Heuristics for understanding vision-language-model training configuration."""

from __future__ import annotations

from math import isfinite

from trainlens.models.analysis import Recommendation, Signal
from trainlens.training_profile import TrainingProfile


def detect_vlm_training_signals(profile: TrainingProfile) -> tuple[Signal, ...]:
    """Return evidence-backed signals derived from a normalized VLM training profile."""

    if not _is_vlm_profile(profile):
        return ()

    signals: list[Signal] = []
    if profile.strategy == "vlm_projector_alignment":
        signals.append(
            Signal(
                title="Projector-only VLM alignment",
                detail=(
                    "The multimodal projector appears trainable while the vision and "
                    "language towers are frozen or unavailable for updates."
                ),
                severity="info",
                evidence=_component_evidence(profile),
            )
        )
    elif profile.strategy == "vlm_adapter_finetune":
        evidence = list(_component_evidence(profile))
        rank = _numeric_parameter(profile, "peft.r")
        targets = profile.parameters.get("peft.target_modules")
        if rank is not None:
            evidence.append(f"lora_rank={rank:g}")
        if isinstance(targets, tuple) and targets:
            evidence.append(f"target_modules={', '.join(targets)}")
        signals.append(
            Signal(
                title="Parameter-efficient VLM fine-tuning",
                detail="TrainLens detected PEFT/LoRA configuration on a vision-language model.",
                severity="info",
                evidence=tuple(evidence),
            )
        )

    rank = _numeric_parameter(profile, "peft.r")
    if rank is not None and rank <= 4:
        signals.append(
            Signal(
                title="Very small VLM adapter rank",
                detail=(
                    "The configured LoRA rank is small for a multimodal adaptation; "
                    "capacity may be intentionally constrained."
                ),
                severity="warning",
                evidence=(f"lora_rank={rank:g}",),
            )
        )

    sequence_key, sequence_length = _first_numeric_parameter(
        profile,
        "training.max_length",
        "training.model_max_length",
    )
    if sequence_key is not None and sequence_length is not None and sequence_length > 0:
        signals.append(
            Signal(
                title="Fixed multimodal sequence length",
                detail=(
                    "A fixed sequence length is configured for VLM training; verify that "
                    "preprocessing does not truncate required image or video tokens."
                ),
                severity="info",
                evidence=(f"{sequence_key}={sequence_length:g}",),
            )
        )

    base_lr = _numeric_parameter(profile, "training.learning_rate")
    projector_lr = _numeric_parameter(profile, "training.mm_projector_lr")
    vision_lr = _numeric_parameter(profile, "training.vision_tower_lr")
    if projector_lr is not None and (
        base_lr is None or abs(projector_lr - base_lr) > max(abs(base_lr) * 1e-12, 1e-15)
    ):
        evidence = [f"mm_projector_lr={projector_lr:g}"]
        if base_lr is not None:
            evidence.append(f"base_lr={base_lr:g}")
        if vision_lr is not None:
            evidence.append(f"vision_tower_lr={vision_lr:g}")
        signals.append(
            Signal(
                title="Multimodal learning-rate split",
                detail="The projector uses a learning rate distinct from the base training rate.",
                severity="info",
                evidence=tuple(evidence),
            )
        )

    frozen = set(profile.frozen_components)
    if (
        not profile.trainable_components
        and {"vision_tower", "language_model", "mm_projector"}.issubset(frozen)
    ):
        signals.append(
            Signal(
                title="All detected VLM components are frozen",
                detail=(
                    "The vision tower, language model, and multimodal projector all appear "
                    "frozen; confirm that an adapter or another trainable module is present."
                ),
                severity="warning",
                evidence=_component_evidence(profile),
            )
        )

    return tuple(signals)


def vlm_training_recommendations(
    profile: TrainingProfile,
    signals: tuple[Signal, ...] | list[Signal],
) -> tuple[Recommendation, ...]:
    """Recommend controlled checks for VLM-specific training configuration signals."""

    titles = {signal.title for signal in signals}
    recommendations: list[Recommendation] = []
    if "All detected VLM components are frozen" in titles:
        recommendations.append(
            Recommendation(
                action="Confirm the intended trainable VLM modules or adapter injection.",
                rationale=(
                    "A run with every detected multimodal component frozen may perform no "
                    "meaningful parameter update."
                ),
                confidence=0.92,
            )
        )
    if "Very small VLM adapter rank" in titles:
        recommendations.append(
            Recommendation(
                action=(
                    "Run one controlled adapter-capacity experiment with a higher LoRA rank "
                    "while keeping data, learning rate, and target modules fixed."
                ),
                rationale=(
                    "A rank comparison isolates whether adapter capacity limits multimodal "
                    "adaptation without changing the rest of the training recipe."
                ),
                confidence=0.72,
            )
        )
    if "Fixed multimodal sequence length" in titles:
        recommendations.append(
            Recommendation(
                action=(
                    "Inspect a preprocessed multimodal batch and verify that visual tokens "
                    "survive the configured sequence-length limit."
                ),
                rationale=(
                    "Truncated visual tokens can look like an optimization problem while the "
                    "actual issue is input construction."
                ),
                confidence=0.82,
            )
        )
    if profile.strategy == "vlm_projector_alignment":
        recommendations.append(
            Recommendation(
                action=(
                    "Validate projector alignment on held-out multimodal examples before "
                    "unfreezing additional towers."
                ),
                rationale=(
                    "Projector-only training is easiest to diagnose before language or vision "
                    "weights introduce additional degrees of freedom."
                ),
                confidence=0.84,
            )
        )
    return tuple(recommendations)


def _is_vlm_profile(profile: TrainingProfile) -> bool:
    return (
        profile.strategy.startswith("vlm_")
        or any(name.startswith("vlm.") for name in profile.parameters)
        or any(
            name in {"vision_tower", "language_model", "mm_projector"}
            for name in (*profile.trainable_components, *profile.frozen_components)
        )
    )


def _component_evidence(profile: TrainingProfile) -> tuple[str, ...]:
    evidence: list[str] = []
    if profile.trainable_components:
        evidence.append(f"trainable={', '.join(profile.trainable_components)}")
    if profile.frozen_components:
        evidence.append(f"frozen={', '.join(profile.frozen_components)}")
    return tuple(evidence)


def _numeric_parameter(profile: TrainingProfile, name: str) -> float | None:
    value = profile.parameters.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    numeric = float(value)
    return numeric if isfinite(numeric) else None


def _first_numeric_parameter(
    profile: TrainingProfile,
    *names: str,
) -> tuple[str | None, float | None]:
    for name in names:
        value = _numeric_parameter(profile, name)
        if value is not None:
            return name, value
    return None, None
