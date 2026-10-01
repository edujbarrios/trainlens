"""Training-session analyzer."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any, cast

from trainlens.analysis_config import AnalysisConfig
from trainlens.analyzers.base import Analyzer
from trainlens.analyzers.metrics import extract_metric_series, paired_metric
from trainlens.analyzers.traces import extract_trace_events
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
from trainlens.heuristics.vlm_training import (
    detect_vlm_training_signals,
    vlm_training_recommendations,
)
from trainlens.introspection.models import ModelCandidate
from trainlens.introspection.selection import framework_source_for_model
from trainlens.models.analysis import AnalysisResult, Recommendation
from trainlens.models.snapshot import NotebookSnapshot
from trainlens.training_profile import TrainingProfile, inspect_training_profile


class TrainingSessionAnalyzer(Analyzer):
    """Combines model, metric, dataset, and feature signals."""

    name = "training_session"

    def analyze(
        self,
        snapshot: NotebookSnapshot,
        model: ModelCandidate | None,
        *,
        config: AnalysisConfig | None = None,
    ) -> AnalysisResult:
        namespace = _namespace_with_framework_artifacts(snapshot)
        if config is not None and config.metrics is not None:
            namespace["trainlens_explicit_metrics"] = config.metrics
        metric_series = extract_metric_series(namespace)
        train_acc, validation_acc = paired_metric(metric_series, "accuracy")
        train_loss, validation_loss = paired_metric(metric_series, "loss")
        framework = model.framework if model else _first_artifact_framework(snapshot)
        model_name = model.display_name if model else _first_artifact_model_name(snapshot)
        model_ref = model.object_ref if model else _first_artifact_model_ref(snapshot)
        trainer = _trainer_for_config(snapshot, config, model_ref)
        training_profile = inspect_training_profile(
            model_ref,
            trainer=trainer,
            namespace=snapshot.raw_namespace,
        )
        result = AnalysisResult(model_name=model_name, framework=framework)
        families = detect_foundation_architecture(model_ref, namespace)

        if model:
            result.summary.append(f"Detected {model.display_name} from `{model.variable_name}`.")
            if model.framework:
                result.summary.append(f"Framework appears to be {model.framework}.")
        elif snapshot.framework_artifacts:
            artifact = snapshot.framework_artifacts[0]
            result.summary.append(
                f"Detected {artifact.framework} training evidence from `{artifact.variable_name}`."
            )
        else:
            result.summary.append("No trained model object was confidently detected.")

        for artifact in snapshot.framework_artifacts:
            if artifact.history or artifact.log_history or artifact.latest_metrics:
                result.summary.append(
                    f"Adapted {artifact.framework} metrics from `{artifact.variable_name}`."
                )
            elif artifact.training_parameters:
                result.summary.append(
                    f"Captured {artifact.framework} training parameters from "
                    f"`{artifact.variable_name}`."
                )

        _append_training_profile_summary(result, training_profile)

        if train_acc and train_acc.last is not None:
            result.metrics["train_accuracy"] = train_acc.last
            if train_acc.delta is not None:
                result.summary.append(
                    f"Training accuracy changed from {train_acc.first:.3f} to {train_acc.last:.3f}."
                )
        if validation_acc and validation_acc.last is not None:
            result.metrics["validation_accuracy"] = validation_acc.last
        if train_loss and train_loss.last is not None:
            result.metrics["train_loss"] = train_loss.last
            if train_loss.first is not None and train_loss.delta is not None:
                result.summary.append(
                    f"Training loss changed from {train_loss.first:.3f} to {train_loss.last:.3f}."
                )
        if validation_loss and validation_loss.last is not None:
            result.metrics["validation_loss"] = validation_loss.last
            if validation_loss.first is not None and validation_loss.delta is not None:
                result.summary.append(
                    "Validation loss changed from "
                    f"{validation_loss.first:.3f} to {validation_loss.last:.3f}."
                )
        if families:
            result.summary.append(f"Foundation-model profile: {', '.join(families).upper()}.")

        result.trace.extend(extract_trace_events(namespace))
        labels = (
            config.labels
            if config is not None and config.labels is not None
            else _first_present(namespace, "y_train", "y", "labels", "target")
        )
        for signal in (
            detect_overfitting(train_acc, validation_acc),
            detect_validation_instability(validation_acc),
            detect_convergence(validation_acc or train_acc),
            detect_class_imbalance(labels),
            detect_loss_plateau(validation_loss or train_loss),
            detect_contrastive_misalignment(metric_series),
            detect_adapter_pressure(namespace),
        ):
            if signal:
                result.signals.append(signal)

        vlm_signals = detect_vlm_training_signals(training_profile)
        result.signals.extend(vlm_signals)

        if model:
            result.top_features = top_features(model.object_ref, infer_feature_names(namespace))

        result.recommendations.extend(self._recommendations(result))
        result.recommendations.extend(foundation_recommendations(families, result.signals))
        result.recommendations.extend(vlm_training_recommendations(training_profile, vlm_signals))
        return result

    def _recommendations(self, result: AnalysisResult) -> list[Recommendation]:
        recommendations: list[Recommendation] = []
        titles = {signal.title for signal in result.signals}
        if "Possible overfitting" in titles:
            recommendations.append(
                Recommendation(
                    action="Tune regularization or depth-related hyperparameters.",
                    rationale=(
                        "The validation gap suggests the model may be memorizing training examples."
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
                        "Inspecting false positives and false negatives usually reveals the next useful experiment."
                    ),
                    confidence=0.52,
                )
            )
        return recommendations


def _trainer_for_config(
    snapshot: NotebookSnapshot,
    config: AnalysisConfig | None,
    model_ref: object | None,
) -> object | None:
    if config is None or config.trainer is None:
        return framework_source_for_model(snapshot, "huggingface", model_ref)
    selector = config.trainer
    if isinstance(selector, str):
        if selector not in snapshot.raw_namespace:
            raise ValueError(f"trainer variable {selector!r} was not found in the notebook snapshot")
        return snapshot.raw_namespace[selector]
    return selector


def _append_training_profile_summary(result: AnalysisResult, profile: TrainingProfile) -> None:
    if not (
        profile.parameters
        or profile.trainable_components
        or profile.frozen_components
        or profile.observations
    ):
        return
    if profile.strategy != "unknown":
        result.summary.append(
            "Training strategy appears to be " + profile.strategy.replace("_", " ") + "."
        )
    if profile.trainable_components:
        result.summary.append(f"Trainable components: {', '.join(profile.trainable_components)}.")
    if profile.frozen_components:
        result.summary.append(f"Frozen components: {', '.join(profile.frozen_components)}.")

    learning_rates = []
    for key, label in (
        ("training.learning_rate", "base"),
        ("training.mm_projector_lr", "projector"),
        ("training.vision_tower_lr", "vision tower"),
    ):
        value = profile.parameters.get(key)
        if value is not None:
            learning_rates.append(f"{label}={value}")
    if learning_rates:
        result.summary.append("Learning rates: " + ", ".join(learning_rates) + ".")

    rank = profile.parameters.get("peft.r")
    alpha = profile.parameters.get("peft.lora_alpha")
    targets = profile.parameters.get("peft.target_modules")
    adapter_parts = []
    if rank is not None:
        adapter_parts.append(f"rank={rank}")
    if alpha is not None:
        adapter_parts.append(f"alpha={alpha}")
    if isinstance(targets, tuple) and targets:
        adapter_parts.append(f"targets={', '.join(targets)}")
    if adapter_parts:
        result.summary.append("PEFT adapter: " + ", ".join(adapter_parts) + ".")


def _first_present(namespace: dict[str, object], *names: str) -> Iterable[Any] | None:
    for name in names:
        value = namespace.get(name)
        if isinstance(value, Iterable) and not isinstance(value, str | bytes | Mapping | Iterator):
            return value
    return None


def _namespace_with_framework_artifacts(snapshot: NotebookSnapshot) -> dict[str, Any]:
    namespace = dict(snapshot.raw_namespace)
    for artifact in snapshot.framework_artifacts:
        prefix = f"{artifact.variable_name}_{artifact.framework}"
        if artifact.history:
            namespace[f"{prefix}_history"] = artifact.history
        if artifact.log_history:
            namespace[f"{prefix}_log_history"] = list(artifact.log_history)
        if artifact.latest_metrics:
            namespace[f"{prefix}_metrics"] = artifact.latest_metrics
    return namespace


def _first_artifact_framework(snapshot: NotebookSnapshot) -> str | None:
    if not snapshot.framework_artifacts:
        return None
    return snapshot.framework_artifacts[0].framework


def _first_artifact_model_name(snapshot: NotebookSnapshot) -> str | None:
    for artifact in snapshot.framework_artifacts:
        if artifact.model_name:
            return artifact.model_name
    return None


def _first_artifact_model_ref(snapshot: NotebookSnapshot) -> object | None:
    for artifact in snapshot.framework_artifacts:
        if artifact.model_ref is not None:
            return cast(object, artifact.model_ref)
    return None
