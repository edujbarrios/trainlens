from __future__ import annotations

import pytest

from trainlens import AnalysisConfig, analyze
from trainlens.analyzers.base import Analyzer
from trainlens.analyzers.registry import register_analyzer, unregister_analyzer
from trainlens.introspection.models import ModelCandidate
from trainlens.models.analysis import AnalysisResult
from trainlens.models.snapshot import NotebookSnapshot


class DummyAnalyzer(Analyzer):
    name = "dummy_test_analyzer"

    def analyze(
        self,
        snapshot: NotebookSnapshot,
        model: ModelCandidate | None,
        *,
        config: AnalysisConfig | None = None,
    ) -> AnalysisResult:
        return AnalysisResult(summary=[f"plugin saw {len(snapshot.variables)} variables"])


def test_registered_analyzer_can_be_selected_from_analysis_config() -> None:
    analyzer = DummyAnalyzer()
    register_analyzer(analyzer)
    try:
        result = analyze({"metric": 1.0}, config=AnalysisConfig(analyzer=analyzer.name))
        assert result.summary == ["plugin saw 1 variables"]

        with pytest.raises(ValueError, match="already registered"):
            register_analyzer(DummyAnalyzer())
        register_analyzer(DummyAnalyzer(), replace=True)
    finally:
        unregister_analyzer(analyzer.name)


def test_analysis_reports_unknown_analyzer_and_protects_builtin() -> None:
    with pytest.raises(ValueError, match="is not registered"):
        analyze({}, config=AnalysisConfig(analyzer="missing_analyzer"))

    with pytest.raises(ValueError, match="cannot be unregistered"):
        unregister_analyzer("training_session")
