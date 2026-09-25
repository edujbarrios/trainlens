"""Shared semantic rules for normalized training metrics."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

MetricDirection = Literal["lower", "higher"]

_LOWER_IS_BETTER = frozenset(
    {
        "loss",
        "error",
        "perplexity",
        "wer",
        "cer",
        "latency",
        "fad",
        "frechet",
        "mae",
        "mape",
        "mse",
        "msle",
        "rmse",
    }
)
_HIGHER_IS_BETTER = frozenset(
    {
        "accuracy",
        "acc",
        "auc",
        "f1",
        "precision",
        "recall",
        "score",
        "map",
        "ndcg",
    }
)
_UNIT_INTERVAL_METRICS = frozenset(
    {"accuracy", "acc", "auc", "f1", "precision", "recall", "map", "ndcg"}
)


@dataclass(frozen=True)
class MetricSemantics:
    """Known optimization direction and optional natural bounds for a metric."""

    direction: MetricDirection
    lower_bound: float | None = None
    upper_bound: float | None = None


def metric_semantics(name: str) -> MetricSemantics | None:
    """Return shared semantics inferred from normalized metric-name tokens."""

    tokens = _metric_tokens(name)
    if tokens.intersection(_LOWER_IS_BETTER):
        return MetricSemantics(direction="lower", lower_bound=0.0)
    if tokens.intersection(_HIGHER_IS_BETTER):
        if tokens.intersection(_UNIT_INTERVAL_METRICS):
            return MetricSemantics(direction="higher", lower_bound=0.0, upper_bound=1.0)
        return MetricSemantics(direction="higher")
    return None


def metric_direction(name: str) -> MetricDirection | None:
    """Return whether a known metric should move lower or higher."""

    semantics = metric_semantics(name)
    return semantics.direction if semantics is not None else None


def metric_bounds(name: str) -> tuple[float | None, float | None]:
    """Return known natural bounds, or ``(None, None)`` for unknown metrics."""

    semantics = metric_semantics(name)
    if semantics is None:
        return None, None
    return semantics.lower_bound, semantics.upper_bound


def _metric_tokens(name: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", name.lower()))
