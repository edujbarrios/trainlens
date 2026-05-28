"""Feature extraction helpers."""

from __future__ import annotations

from collections.abc import Sequence


def infer_feature_names(namespace: dict[str, object]) -> list[str]:
    for name in ("feature_names", "features", "columns"):
        value = namespace.get(name)
        if isinstance(value, Sequence) and not isinstance(value, str | bytes):
            return [str(item) for item in value]
    for name in ("X_train", "X", "train_X"):
        value = namespace.get(name)
        columns = getattr(value, "columns", None)
        if columns is not None:
            return [str(item) for item in columns]
    return []


def top_features(model: object, feature_names: list[str], limit: int = 5) -> list[str]:
    weights = getattr(model, "feature_importances_", None)
    if weights is None:
        coef = getattr(model, "coef_", None)
        weights = coef[0] if getattr(coef, "ndim", 1) > 1 else coef
    if weights is None or not feature_names:
        return []
    ranked = sorted(
        zip(feature_names, weights, strict=False),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    )
    return [name for name, _ in ranked[:limit]]
