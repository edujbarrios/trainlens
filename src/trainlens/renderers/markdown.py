"""Markdown rendering for notebook output."""

from __future__ import annotations

from trainlens.models.analysis import AnalysisResult


class MarkdownRenderer:
    """Render analysis results into notebook-friendly Markdown."""

    def render(self, result: AnalysisResult) -> str:
        lines: list[str] = []
        model = result.model_name or "No confident model detected"
        lines.append(f"## TrainLens Report\n\n**Model detected:** {model}")
        if result.framework:
            lines.append(f"\n**Framework:** {result.framework}")
        if result.summary:
            lines.append("\n### Training summary")
            lines.extend(f"- {item}" for item in result.summary)
        explanation = _result_explanation(result)
        if explanation:
            lines.append("\n### Result explanation")
            lines.extend(f"- {item}" for item in explanation)
        if result.metrics:
            lines.extend(
                [
                    "\n### Metrics",
                    "| Metric | Value |",
                    "| --- | ---: |",
                ]
            )
            lines.extend(
                f"| {name} | {value:.3f} |" for name, value in sorted(result.metrics.items())
            )
        if result.trace:
            lines.extend(
                [
                    "\n### Execution trace",
                    "| Step | Epoch | Event | Metrics |",
                    "| ---: | ---: | --- | --- |",
                ]
            )
            for event in result.trace:
                step = str(event.step) if event.step is not None else ""
                epoch = f"{event.epoch:.2f}" if event.epoch is not None else ""
                label = event.name or event.message or "training event"
                metrics = (
                    ", ".join(f"{name}={value:.3f}" for name, value in event.metrics.items())
                    or "none"
                )
                lines.append(f"| {step} | {epoch} | {label} | {metrics} |")
        if result.signals:
            lines.append("\n### Potential issues")
            for signal in result.signals:
                lines.append(f"- **{signal.title}:** {signal.detail}")
        if result.top_features:
            lines.append("\n### Top features")
            lines.extend(
                f"{index}. {name}" for index, name in enumerate(result.top_features, start=1)
            )
        if result.recommendations:
            lines.append("\n### Recommended next steps")
            lines.extend(f"- {item.action} _{item.rationale}_" for item in result.recommendations)
            lines.append("\n### Improvement plan")
            for index, item in enumerate(
                sorted(result.recommendations, key=lambda value: value.confidence, reverse=True),
                start=1,
            ):
                confidence = int(item.confidence * 100)
                lines.append(
                    f"{index}. {item.action} Confidence: {confidence}%. "
                    f"Why: {item.rationale}"
                )
        return "\n".join(lines).strip() + "\n"


def _result_explanation(result: AnalysisResult) -> list[str]:
    explanation: list[str] = []
    train_loss = result.metrics.get("train_loss")
    validation_loss = result.metrics.get("validation_loss")
    if train_loss is not None and validation_loss is not None:
        gap = validation_loss - train_loss
        if gap > 0.25:
            explanation.append(
                "Validation loss is materially higher than training loss, so the run may be "
                "generalizing worse than it fits the training batches."
            )
        elif gap < -0.05:
            explanation.append(
                "Validation loss is lower than training loss, which can happen with strong "
                "regularization, easier validation data, or noisy training batches."
            )
        else:
            explanation.append(
                "Training and validation loss are close, so the current run does not show a "
                "large loss gap."
            )
    train_accuracy = result.metrics.get("train_accuracy")
    validation_accuracy = result.metrics.get("validation_accuracy")
    if train_accuracy is not None and validation_accuracy is not None:
        gap = train_accuracy - validation_accuracy
        if gap > 0.12:
            explanation.append(
                "Training accuracy is much higher than validation accuracy, which points to "
                "possible overfitting."
            )
        else:
            explanation.append(
                "Training and validation accuracy are reasonably aligned for this snapshot."
            )
    if result.signals:
        warning_count = sum(signal.severity != "info" for signal in result.signals)
        explanation.append(
            f"TrainLens found {len(result.signals)} signal(s), including "
            f"{warning_count} warning or critical item(s)."
        )
    if result.trace:
        explanation.append(
            "Execution trace evidence is available, so the metric readings can be tied back "
            "to concrete training events."
        )
    return explanation
