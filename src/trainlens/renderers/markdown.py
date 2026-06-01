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
        return "\n".join(lines).strip() + "\n"
