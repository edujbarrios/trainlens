"""High-level analysis pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trainlens.analyzers.registry import default_registry
from trainlens.introspection import NotebookInspector
from trainlens.models.analysis import AnalysisResult
from trainlens.models.snapshot import NotebookSnapshot


def snapshot_namespace(namespace: Mapping[str, Any]) -> NotebookSnapshot:
    """Capture one notebook snapshot for reuse across analysis/reporting paths."""

    return NotebookInspector().snapshot(namespace)


def analyze_snapshot(snapshot: NotebookSnapshot) -> AnalysisResult:
    """Analyze an already captured notebook snapshot."""

    inspector = NotebookInspector()
    model = next(iter(inspector.find_models(snapshot)), None)
    analyzer = default_registry().get("training_session")
    return analyzer.analyze(snapshot, model)


def explain_namespace(namespace: Mapping[str, Any]) -> AnalysisResult:
    """Analyze a notebook namespace using built-in heuristics."""

    return analyze_snapshot(snapshot_namespace(namespace))
