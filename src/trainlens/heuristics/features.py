"""Feature extraction helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast


def infer_feature_names(namespace: dict[str, object]) -> list[str]:
    for name in ("feature_names", "features", "columns"):
        value = namespace.get(name)
        if isinstance(value, Sequence) and not isinstance(value, str | bytes):
            try:
                return [str(item) for item in value]
            except Exception:
                continue
    for name in ("X_train", "X", "train_X"):
        value = namespace.get(name)
        columns = _safe_getattr(value, "columns")
        if columns is not None:
            try:
                return [str(item) for item in columns]
            except Exception:
                continue
    return []


def top_features(model: object, feature_names: list[str], limit: int = 5) -> list[str]:
    weights = _safe_getattr(model, "feature_importances_")
    if weights is None:
        coef = _safe_getattr(model, "coef_")
        if coef is not None:
            ndim = _safe_getattr(coef, "ndim", 1)
            try:
                weights = coef[0] if ndim > 1 else coef
            except Exception:
                return []
    if weights is None or not feature_names:
        return []
    weights = cast(Sequence[Any], weights)
    numeric: list[tuple[str, float]] = []
    try:
        pairs = zip(feature_names, weights, strict=False)
        for name, weight in pairs:
            try:
                numeric.append((name, abs(float(weight))))
            except (TypeError, ValueError, OverflowError):
                continue
    except Exception:
        return []
    numeric.sort(key=lambda item: item[1], reverse=True)
    return [name for name, _ in numeric[:limit]]


def _safe_getattr(value: object | None, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    try:
        return getattr(value, name, default)
    except Exception:
        return default
