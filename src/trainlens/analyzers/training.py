"""Training-session analyzer."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from trainlens.analyzers.base import Analyzer
from trainlens.analyzers.metrics import extract_metric_series, paired_metric
from trainlens.heuristics import (
    detect_class_imbalance,
    detect_convergence,
    detect_overfitting,
    detect_validation_instability,
)
from trainlens.heuristics.features import infer_feature_names, top_features
from trainlens.heuristics.foundation import (
    detect_adapter_pressure,
    detect_contrastive_misalignment,
    detect_foundation_architecture,
    detect_loss_plateau,
    foundation_recommendations,
)
from trainlens.introspection.models import ModelCandidate
from trainlens.models.analysis import AnalysisResult, Recommendation
from trainlens.models.snapshot import NotebookSnapshot


class TrainingSessionAnalyzer(Analyzer):
    """Combines model, metric, dataset, and feature signals."""

    name = "training_session"

    def analyze(self, snapshot: NotebookSnapshot, model: ModelCandidate | None) -> AnalysisResult:
        namespace = snapshot.raw_namespace
        metric_series = extract_metric_series(namespace)
        train_acc, validation_acc = paired_metric(metric_series, "accuracy")
        train_loss, validation_loss = paired_metric(metric_series, "loss")
        result = AnalysisResult(
            model_name=model.display_name if model else None,
            framework=model.framework if model else None,
        )
        families = detect_foundation_architecture(model.object_ref if model else None, namespace)

        if model:
            result.summary.append(f"Detected {model.display_name} from `{model.variable_name}`.")
            if model.framework:
                result.summary.append(f"Framework appears to be {model.framework}.")
        else:
            result.summary.append("No trained model object was confidently detected.")

        if train_acc and train_acc.delta is not None:
            result.summary.append(
                f"Training accuracy changed from {train_acc.first:.3f} to {train_acc.last:.3f}."
            )
            result.metrics["train_accuracy"] = train_acc.last or 0.0
        if validation_acc and validation_acc.last is not None:
            result.metrics["validation_accuracy"] = validation_acc.last
        if train_loss and train_loss.last is not None:
            result.metrics["train_loss"] = train_loss.last
        if validation_loss and validation_loss.last is not None:
            result.metrics["validation_loss"] = validation_loss.last
        if families:
            result.summary.append(f"Foundation-model profile: {', '.join(families).upper()}.")

        for signal in (
            detect_overfitting(train_acc, validation_acc),
            detect_validation_instability(validation_acc),
            detect_convergence(validation_acc or train_acc),
            detect_class_imbalance(_first_present(namespace, "y_train", "y", "labels", "target")),
            detect_loss_plateau(validation_loss or train_loss),
            detect_contrastive_misalignment(metric_series),
            detect_adapter_pressure(namespace),
        ):
            if signal:
                result.signals.append(signal)

        if model:
            result.top_features = top_features(model.object_ref, infer_feature_names(namespace))

        result.recommendations.extend(self._recommendations(result))
        result.recommendations.extend(foundation_recommendations(families, result.signals))
        return result

    def _recommendations(self, result: AnalysisResult) -> list[Recommendation]:
        recommendations: list[Recommendation] = []
        titles = {signal.title for signal in result.signals}
        if "Possible overfitting" in titles:
            recommendations.append(
                Recommendation(
                    action="Tune regularization or depth-related hyperparameters.",
                    rationale=(
                        "The validation gap suggests the model may be memorizing "
                        "training examples."
                    ),
                    confidence=0.78,
                )
            )
        if "Class imbalance detected" in titles:
            recommendations.append(
                Recommendation(
                    action="Try stratified splitting, class weights, or resampling.",
                    rationale="Minority-class performance can be hidden by aggregate accuracy.",
                    confidence=0.74,
                )
            )
        if not recommendations:
            recommendations.append(
                Recommendation(
                    action="Run a focused validation error analysis.",
                    rationale=(
                        "Inspecting false positives and false negatives usually "
                        "reveals the next useful experiment."
                    ),
                    confidence=0.52,
                )
            )
        return recommendations


def _first_present(namespace: dict[str, object], *names: str) -> Iterable[Any] | None:
    for name in names:
        value = namespace.get(name)
        if isinstance(value, Iterable):
            return value
    return None
