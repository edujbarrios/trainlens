"""Analyzer registration."""

from __future__ import annotations

from collections.abc import Iterable

from trainlens.analyzers.base import Analyzer


class AnalyzerRegistry:
    """Small extension point for built-in and plugin analyzers."""

    def __init__(self, analyzers: Iterable[Analyzer] | None = None) -> None:
        self._analyzers: dict[str, Analyzer] = {}
        for analyzer in analyzers or ():
            self.register(analyzer)

    def register(self, analyzer: Analyzer) -> None:
        self._analyzers[analyzer.name] = analyzer

    def get(self, name: str) -> Analyzer:
        return self._analyzers[name]

    def all(self) -> tuple[Analyzer, ...]:
        return tuple(self._analyzers.values())


def default_registry() -> AnalyzerRegistry:
    from trainlens.analyzers.training import TrainingSessionAnalyzer

    return AnalyzerRegistry([TrainingSessionAnalyzer()])
