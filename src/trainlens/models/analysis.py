"""Analysis output models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from trainlens.models.trace import TraceEvent

Severity = Literal["info", "warning", "critical"]


@dataclass(frozen=True)
class Signal:
    """A detected training behavior or risk."""

    title: str
    detail: str
    severity: Severity = "info"
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class Recommendation:
    """A concrete next step suggested by an analyzer."""

    action: str
    rationale: str
    confidence: float = 0.5


@dataclass
class AnalysisResult:
    """Aggregated result produced by the training analysis pipeline."""

    model_name: str | None = None
    framework: str | None = None
    summary: list[str] = field(default_factory=list)
    signals: list[Signal] = field(default_factory=list)
    recommendations: list[Recommendation] = field(default_factory=list)
    top_features: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    trace: list[TraceEvent] = field(default_factory=list)

    def has_findings(self) -> bool:
        return bool(
            self.summary
            or self.signals
            or self.recommendations
            or self.top_features
            or self.trace
        )
