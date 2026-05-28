"""Heuristic analysis rules."""

from trainlens.heuristics.balance import detect_class_imbalance
from trainlens.heuristics.foundation import (
    detect_adapter_pressure,
    detect_contrastive_misalignment,
    detect_foundation_architecture,
    detect_loss_plateau,
    foundation_recommendations,
)
from trainlens.heuristics.metrics import (
    detect_convergence,
    detect_overfitting,
    detect_validation_instability,
)

__all__ = [
    "detect_adapter_pressure",
    "detect_class_imbalance",
    "detect_convergence",
    "detect_contrastive_misalignment",
    "detect_foundation_architecture",
    "detect_loss_plateau",
    "detect_overfitting",
    "detect_validation_instability",
    "foundation_recommendations",
]
