"""Simple notebook-local run storage."""

from __future__ import annotations

from copy import deepcopy

from trainlens.comparison import compare_runs, render_run_comparison
from trainlens.models.analysis import AnalysisResult


class InMemoryRunStore:
    """Keeps recent reports for a notebook session."""

    def __init__(self, max_runs: int | None = None) -> None:
        if max_runs is not None and (
            isinstance(max_runs, bool) or not isinstance(max_runs, int)
        ):
            raise TypeError("max_runs must be an integer or None")
        if max_runs is not None and max_runs < 1:
            msg = "max_runs must be at least 1 when provided"
            raise ValueError(msg)
        self.max_runs = max_runs
        self._runs: list[AnalysisResult] = []
        self._names: list[str | None] = []

    @property
    def runs(self) -> tuple[AnalysisResult, ...]:
        return tuple(self._runs)

    @property
    def names(self) -> tuple[str | None, ...]:
        """Return stable user-assigned run labels in capture order."""

        return tuple(self._names)

    def capture(self, result: AnalysisResult, *, name: str | None = None) -> None:
        normalized_name = self._validate_name(name)
        self._runs.append(deepcopy(result))
        self._names.append(normalized_name)
        if self.max_runs is not None:
            self._runs = self._runs[-self.max_runs :]
            self._names = self._names[-self.max_runs :]

    def latest(self) -> AnalysisResult | None:
        return self._runs[-1] if self._runs else None

    def clear(self) -> None:
        self._runs.clear()
        self._names.clear()

    def get(self, selector: str | int) -> AnalysisResult:
        """Return a captured run by stable name or one-based capture index."""

        index = self._resolve_index(selector)
        return self._runs[index]

    def render_comparison(
        self,
        baseline: str | int | None = None,
        experiment: str | int | None = None,
    ) -> str:
        if not self._runs:
            return (
                "## TrainLens Run Comparison\n\n"
                "No runs captured yet. Use `%explain_training` first.\n"
            )
        if baseline is None and experiment is None:
            if len(self._runs) >= 2:
                return self._render_pair(
                    len(self._runs) - 2,
                    len(self._runs) - 1,
                    baseline_name="previous run",
                    experiment_name="latest run",
                )
            return self._render_single_run_table()
        if baseline is None or experiment is None:
            raise ValueError(
                "Provide both baseline and experiment selectors, or neither for the latest two."
            )
        baseline_index = self._resolve_index(baseline)
        experiment_index = self._resolve_index(experiment)
        return self._render_pair(
            baseline_index,
            experiment_index,
            baseline_name=self._display_name(baseline_index),
            experiment_name=self._display_name(experiment_index),
        )

    def _render_pair(
        self,
        baseline_index: int,
        experiment_index: int,
        *,
        baseline_name: str,
        experiment_name: str,
    ) -> str:
        baseline = self._runs[baseline_index]
        experiment = self._runs[experiment_index]
        comparison = compare_runs(
            baseline,
            experiment,
            baseline_name=baseline_name,
            experiment_name=experiment_name,
        )
        markdown = render_run_comparison(comparison).rstrip()
        metadata = self._render_metadata(baseline_name, baseline, experiment_name, experiment)
        return f"{markdown}\n\n{metadata}\n"

    def _render_metadata(
        self,
        baseline_name: str,
        baseline: AnalysisResult,
        experiment_name: str,
        experiment: AnalysisResult,
    ) -> str:
        lines = [
            "### Run metadata",
            "| Run | Model | Framework | Signals |",
            "| --- | --- | --- | --- |",
        ]
        for name, result in ((baseline_name, baseline), (experiment_name, experiment)):
            signals = ", ".join(signal.title for signal in result.signals) or "none"
            lines.append(
                f"| {name} | {result.model_name or 'unknown'} | "
                f"{result.framework or 'unknown'} | {signals} |"
            )
        return "\n".join(lines)

    def _render_single_run_table(self) -> str:
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

    def _validate_name(self, name: str | None) -> str | None:
        if name is None:
            return None
        normalized = name.strip()
        if not normalized:
            raise ValueError("run name cannot be empty")
        if normalized in (existing for existing in self._names if existing is not None):
            raise ValueError(f"run name {normalized!r} is already captured")
        return normalized

    def _resolve_index(self, selector: str | int) -> int:
        if isinstance(selector, bool):
            raise TypeError("run selector must be a name or one-based integer index")
        if isinstance(selector, int):
            return self._index_from_one_based(selector)
        if not isinstance(selector, str):
            raise TypeError("run selector must be a name or one-based integer index")
        normalized = selector.strip()
        if not normalized:
            raise ValueError("run selector cannot be empty")
        for index, name in enumerate(self._names):
            if name == normalized:
                return index
        try:
            numeric = int(normalized)
        except ValueError as exc:
            available = ", ".join(name for name in self._names if name) or "none"
            raise ValueError(
                f"unknown run {normalized!r}; named runs: {available}. "
                "You can also use a one-based run index."
            ) from exc
        return self._index_from_one_based(numeric)

    def _index_from_one_based(self, selector: int) -> int:
        if selector < 1 or selector > len(self._runs):
            raise ValueError(
                f"run index must be between 1 and {len(self._runs)}; got {selector}"
            )
        return selector - 1

    def _display_name(self, index: int) -> str:
        return self._names[index] or f"run {index + 1}"
