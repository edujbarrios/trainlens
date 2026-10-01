"""Training run metadata."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TypeAlias
from uuid import uuid4

from trainlens.models.metric import MetricSeries

RunValue: TypeAlias = str | int | float | bool | None


@dataclass(frozen=True)
class TrainingRun:
    """A portable snapshot of one completed training run."""

    run_id: str = field(default_factory=lambda: uuid4().hex)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    model_name: str | None = None
    framework: str | None = None
    metrics: tuple[MetricSeries, ...] = ()
    parameters: Mapping[str, RunValue] = field(default_factory=dict)
    metadata: Mapping[str, RunValue] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
