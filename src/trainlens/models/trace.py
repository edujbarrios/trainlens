"""Execution traces captured from training loops."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TraceEvent:
    """One observable event from a training run."""

    step: int | None = None
    epoch: float | None = None
    name: str | None = None
    message: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
