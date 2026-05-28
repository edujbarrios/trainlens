"""Dataset balance heuristics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from trainlens.models.analysis import Signal


def detect_class_imbalance(labels: Iterable[Any] | None) -> Signal | None:
    if labels is None:
        return None
    try:
        counts = Counter(labels)
    except TypeError:
        return None
    if len(counts) < 2:
        return None
    total = sum(counts.values())
    smallest = min(counts.values())
    ratio = smallest / total if total else 1.0
    if ratio < 0.2:
        return Signal(
            title="Class imbalance detected",
            detail=f"The smallest class represents {ratio:.1%} of observed labels.",
            severity="warning",
            evidence=tuple(f"{label}: {count}" for label, count in counts.most_common()),
        )
    return None
