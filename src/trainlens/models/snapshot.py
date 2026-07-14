"""Notebook state snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class VariableInfo:
    """A compact description of one notebook variable."""

    name: str
    type_name: str
    module: str | None = None
    shape: tuple[int | None, ...] | None = None
    length: int | None = None
    value: Any | None = None


@dataclass(frozen=True)
class FrameworkArtifact:
    """Structured training evidence extracted from a known framework object."""

    variable_name: str
    framework: str
    type_name: str
    history: dict[str, tuple[float, ...]]
    log_history: tuple[dict[str, float | int], ...] = ()
    latest_metrics: dict[str, float] | None = None
    training_parameters: dict[str, Any] = field(default_factory=dict)
    model_name: str | None = None
    model_ref: Any | None = None
    confidence: float = 0.75
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class NotebookSnapshot:
    """Serializable-ish view of notebook globals relevant to training analysis."""

    variables: tuple[VariableInfo, ...] = ()
    framework_artifacts: tuple[FrameworkArtifact, ...] = ()
    raw_namespace: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def by_name(self, name: str) -> VariableInfo | None:
        for variable in self.variables:
            if variable.name == name:
                return variable
        return None
