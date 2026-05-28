"""Framework detection without importing heavyweight ML libraries."""

from __future__ import annotations

KNOWN_FRAMEWORK_PREFIXES: dict[str, str] = {
    "accelerate.": "accelerate",
    "diffusers.": "diffusers",
    "peft.": "peft",
    "sklearn.": "sklearn",
    "timm.": "timm",
    "transformers.": "transformers",
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
    if hasattr(obj, "config") and hasattr(obj.config, "model_type"):
        reasons.append("has transformer config")
    if hasattr(obj, "vision_model") or hasattr(obj, "text_model"):
        reasons.append("has multimodal towers")
    if hasattr(obj, "language_model") and hasattr(obj, "vision_tower"):
        reasons.append("looks like VLM")
    if any(hasattr(obj, attr) for attr in ("lora_config", "peft_config", "active_adapters")):
        reasons.append("has adapter fine-tuning state")
    return bool(reasons), tuple(reasons)
