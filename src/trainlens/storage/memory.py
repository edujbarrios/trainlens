"""Simple notebook-local run storage."""

from __future__ import annotations

from trainlens.models.analysis import AnalysisResult


class InMemoryRunStore:
    """Keeps recent reports for a notebook session."""

    def __init__(self) -> None:
        self._runs: list[AnalysisResult] = []

    def capture(self, result: AnalysisResult) -> None:
        self._runs.append(result)

    def render_comparison(self) -> str:
        if not self._runs:
            return (
                "## TrainLens Run Comparison\n\n"
                "No runs captured yet. Use `%explain_training` first.\n"
            )
        lines = [
            "## TrainLens Run Comparison",
            "",
            "| Run | Model | Metrics | Signals |",
            "| --- | --- | --- | --- |",
        ]
        for index, result in enumerate(self._runs, start=1):
            metrics = (
                ", ".join(f"{key}={value:.3f}" for key, value in result.metrics.items())
                or "none"
            )
            signals = ", ".join(signal.title for signal in result.signals) or "none"
            lines.append(f"| {index} | {result.model_name or 'unknown'} | {metrics} | {signals} |")
        return "\n".join(lines) + "\n"
