"""Simple notebook-local run storage."""

from __future__ import annotations

from trainlens.comparison import compare_runs, render_run_comparison
from trainlens.models.analysis import AnalysisResult


class InMemoryRunStore:
    """Keeps recent reports for a notebook session."""

    def __init__(self, max_runs: int | None = None) -> None:
        if max_runs is not None and max_runs < 1:
            msg = "max_runs must be at least 1 when provided"
            raise ValueError(msg)
        self.max_runs = max_runs
        self._runs: list[AnalysisResult] = []

    @property
    def runs(self) -> tuple[AnalysisResult, ...]:
        return tuple(self._runs)

    def capture(self, result: AnalysisResult) -> None:
        self._runs.append(result)
        if self.max_runs is not None:
            self._runs = self._runs[-self.max_runs :]

    def latest(self) -> AnalysisResult | None:
        return self._runs[-1] if self._runs else None

    def clear(self) -> None:
        self._runs.clear()

    def render_comparison(self) -> str:
        if not self._runs:
            return (
                "## TrainLens Run Comparison\n\n"
                "No runs captured yet. Use `%explain_training` first.\n"
            )
        if len(self._runs) >= 2:
            comparison = compare_runs(
                self._runs[-2],
                self._runs[-1],
                baseline_name="previous run",
                experiment_name="latest run",
            )
            return render_run_comparison(comparison)
        lines = [
            "## TrainLens Run Comparison",
            "",
            "Capture at least two runs to compare metric changes.",
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
