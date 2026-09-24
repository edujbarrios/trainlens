"""Dataset balance heuristics."""

from __future__ import annotations

from collections.abc import Iterable
from collections import Counter
from typing import Any

from trainlens.models.analysis import Signal


_MIN_BALANCED_EXPECTATION_RATIO = 0.5


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
    if not total:
        return None
    smallest = min(counts.values())
    class_count = len(counts)
    observed_share = smallest / total
    expected_balanced_count = total / class_count
    balanced_expectation_ratio = smallest / expected_balanced_count
    if balanced_expectation_ratio < _MIN_BALANCED_EXPECTATION_RATIO:
        return Signal(
            title="Class imbalance detected",
            detail=(
                f"The smallest class represents {observed_share:.1%} of observed labels, "
                f"or {balanced_expectation_ratio:.1%} of its balanced expectation "
                f"across {class_count} classes."
            ),
            severity="warning",
            evidence=tuple(f"{label}: {count}" for label, count in counts.most_common()),
        )
    return None
