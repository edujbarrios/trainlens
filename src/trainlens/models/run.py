"""Training run metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from trainlens.models.metric import MetricSeries


@dataclass(frozen=True)
class TrainingRun:
    """A captured notebook training run."""

    run_id: str = field(default_factory=lambda: uuid4().hex)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    model_name: str | None = None
    framework: str | None = None
    metrics: tuple[MetricSeries, ...] = ()
    notes: tuple[str, ...] = ()
