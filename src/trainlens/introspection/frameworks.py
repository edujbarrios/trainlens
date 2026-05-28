"""Framework detection without importing heavyweight ML libraries."""

from __future__ import annotations

KNOWN_FRAMEWORK_PREFIXES: dict[str, str] = {
    "sklearn.": "sklearn",
    "xgboost.": "xgboost",
    "lightgbm.": "lightgbm",
    "torch.": "pytorch",
    "tensorflow.": "tensorflow",
    "keras.": "tensorflow",
}


def detect_framework(obj: object) -> str | None:
    module = getattr(obj.__class__, "__module__", "") or ""
    for prefix, framework in KNOWN_FRAMEWORK_PREFIXES.items():
        if module.startswith(prefix):
            return framework
    return None


def looks_like_model(obj: object) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    if callable(getattr(obj, "fit", None)):
        reasons.append("has fit()")
    if callable(getattr(obj, "predict", None)):
        reasons.append("has predict()")
    if hasattr(obj, "score"):
        reasons.append("has score()")
    if hasattr(obj, "feature_importances_") or hasattr(obj, "coef_"):
        reasons.append("has learned feature weights")
    if hasattr(obj, "state_dict") and callable(getattr(obj, "parameters", None)):
        reasons.append("looks like torch module")
    return bool(reasons), tuple(reasons)
