"""Analyzer interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

from trainlens.introspection.models import ModelCandidate
from trainlens.models.analysis import AnalysisResult
from trainlens.models.snapshot import NotebookSnapshot


class Analyzer(ABC):
    """Base contract for TrainLens analyzers."""

    name: str

    @abstractmethod
    def analyze(self, snapshot: NotebookSnapshot, model: ModelCandidate | None) -> AnalysisResult:
        """Analyze a notebook snapshot."""
