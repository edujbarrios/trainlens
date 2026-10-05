"""Analyzer registration."""

from __future__ import annotations

from collections.abc import Iterable

from trainlens.analyzers.base import Analyzer


class AnalyzerRegistry:
    """Process-local extension point for built-in and plugin analyzers."""

    def __init__(self, analyzers: Iterable[Analyzer] | None = None) -> None:
        self._analyzers: dict[str, Analyzer] = {}
        for analyzer in analyzers or ():
            self.register(analyzer)

    def register(self, analyzer: Analyzer, *, replace: bool = False) -> None:
        name = getattr(analyzer, "name", "")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("analyzer.name must be a non-empty string")
        normalized = name.strip()
        if normalized in self._analyzers and not replace:
            raise ValueError(f"analyzer {normalized!r} is already registered")
        self._analyzers[normalized] = analyzer

    def unregister(self, name: str) -> None:
        if name == "training_session":
            raise ValueError("the built-in training_session analyzer cannot be unregistered")
        self._analyzers.pop(name, None)

    def get(self, name: str) -> Analyzer:
        return self._analyzers[name]

    def all(self) -> tuple[Analyzer, ...]:
        return tuple(self._analyzers.values())


_DEFAULT_REGISTRY: AnalyzerRegistry | None = None


def default_registry() -> AnalyzerRegistry:
    """Return the process-wide analyzer registry used by the analysis pipeline."""

    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        from trainlens.analyzers.training import TrainingSessionAnalyzer

        _DEFAULT_REGISTRY = AnalyzerRegistry([TrainingSessionAnalyzer()])
    return _DEFAULT_REGISTRY


def register_analyzer(analyzer: Analyzer, *, replace: bool = False) -> None:
    """Register a custom analyzer for subsequent :func:`trainlens.analyze` calls."""

    default_registry().register(analyzer, replace=replace)


def unregister_analyzer(name: str) -> None:
    """Remove a previously registered custom analyzer."""

    default_registry().unregister(name)
