"""High-level analysis pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trainlens.analyzers.registry import default_registry
from trainlens.introspection import NotebookInspector
from trainlens.models.analysis import AnalysisResult


def explain_namespace(namespace: Mapping[str, Any]) -> AnalysisResult:
    """Analyze a notebook namespace using built-in heuristics."""

    inspector = NotebookInspector()
    snapshot = inspector.snapshot(namespace)
    model = next(iter(inspector.find_models(snapshot)), None)
    analyzer = default_registry().get("training_session")
    return analyzer.analyze(snapshot, model)
