"""Explicit analysis selection and evidence overrides."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AnalysisConfig:
    """Control which notebook objects TrainLens analyzes.

    ``model`` and ``trainer`` may be either notebook variable names or concrete
    objects. ``metrics`` and ``labels`` let callers provide authoritative
    evidence without relying on notebook-name inference. Set ``strict=True`` to
    reject ambiguous automatic model selection instead of silently choosing the
    highest-confidence candidate.
    """

    model: str | object | None = None
    trainer: str | object | None = None
    metrics: Mapping[str, Any] | None = None
    labels: Iterable[Any] | None = None
    strict: bool = False
