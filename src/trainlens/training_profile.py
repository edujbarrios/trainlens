"""Training-configuration introspection for model fine-tuning workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

TrainingStrategy = Literal[
    "unknown",
    "full_finetune",
    "adapter_finetune",
    "vlm_projector_alignment",
    "vlm_adapter_finetune",
    "vlm_full_finetune",
]
TrainingParameter: TypeAlias = str | int | float | bool | tuple[str, ...] | None

_TRAINING_ARGUMENTS = (
    "learning_rate",
    "weight_decay",
    "warmup_ratio",
    "warmup_steps",
    "lr_scheduler_type",
    "num_train_epochs",
    "max_steps",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "gradient_checkpointing",
    "max_grad_norm",
    "bf16",
    "fp16",
    "tf32",
    "max_length",
    "model_max_length",
    "mm_projector_lr",
    "vision_tower_lr",
)
_VLM_CONFIG_ARGUMENTS = (
    "model_type",
    "mm_projector_type",
    "mm_vision_select_layer",
    "mm_vision_select_feature",
    "vision_feature_layer",
    "vision_feature_select_strategy",
    "image_aspect_ratio",
    "mm_patch_merge_type",
    "mm_use_im_start_end",
    "mm_use_im_patch_token",
    "image_seq_length",
)
_PEFT_ARGUMENTS = (
    "r",
    "lora_alpha",
    "lora_dropout",
    "target_modules",
    "bias",
    "use_rslora",
    "use_dora",
)
_NAMESPACE_ALIASES = {
    "lora_r": "peft.r",
    "lora_rank": "peft.r",
    "lora_alpha": "peft.lora_alpha",
    "lora_dropout": "peft.lora_dropout",
    "target_modules": "peft.target_modules",
    "mm_projector_lr": "training.mm_projector_lr",
    "vision_tower_lr": "training.vision_tower_lr",
    "tune_mm_mlp_adapter": "vlm.tune_mm_mlp_adapter",
    "freeze_mm_mlp_adapter": "vlm.freeze_mm_mlp_adapter",
}
_VLM_MODEL_TYPE_TOKENS = (
    "llava",
    "blip",
    "flamingo",
    "paligemma",
    "vision2seq",
    "qwen2_vl",
    "qwen3_vl",
    "idefics",
    "vlm",
)


@dataclass(frozen=True)
class TrainingProfile:
    """Normalized view of how a model appears to be fine-tuned."""

    strategy: TrainingStrategy
    parameters: Mapping[str, TrainingParameter] = field(default_factory=dict)
    trainable_components: tuple[str, ...] = ()
    frozen_components: tuple[str, ...] = ()
    observations: tuple[str, ...] = ()


def inspect_training_profile(
    model: object | None = None,
    *,
    trainer: object | None = None,
    namespace: Mapping[str, Any] | None = None,
) -> TrainingProfile:
    """Inspect training configuration without importing optional ML frameworks.

    The function relies on safe attribute inspection and duck typing so it can
    understand common Hugging Face, PEFT, and VLM object shapes while keeping
    TrainLens dependency-light.
    """

    if model is None and trainer is not None:
        model = _safe_getattr(trainer, "model")

    parameters: dict[str, TrainingParameter] = {}
    args = _safe_getattr(trainer, "args") if trainer is not None else None
    _extract_attributes(args, _TRAINING_ARGUMENTS, "training", parameters)

    config = _safe_getattr(model, "config")
    _extract_attributes(config, _VLM_CONFIG_ARGUMENTS, "vlm", parameters)

    if namespace is not None:
        for name in _TRAINING_ARGUMENTS:
            if name in namespace and f"training.{name}" not in parameters:
                normalized = _normalize_parameter(namespace[name])
                if normalized is not None:
                    parameters[f"training.{name}"] = normalized
        for source, target in _NAMESPACE_ALIASES.items():
            if source in namespace and target not in parameters:
                normalized = _normalize_parameter(namespace[source])
                if normalized is not None:
                    parameters[target] = normalized

    has_peft = _extract_peft_parameters(model, parameters)
    components = _component_states(model)
    is_vlm = _looks_like_vlm(model, config, components)

    trainable = tuple(name for name, state in components.items() if state is True)
    frozen = tuple(name for name, state in components.items() if state is False)
    strategy = _strategy(
        is_vlm=is_vlm,
        has_peft=has_peft,
        model_state=_component_trainable_state(model),
        components=components,
    )

    observations: list[str] = []
    if is_vlm:
        observations.append("Detected vision-language model training configuration.")
    if has_peft:
        observations.append("Detected parameter-efficient adapter configuration.")
    if trainable:
        observations.append(f"Trainable components: {', '.join(trainable)}.")
    if frozen:
        observations.append(f"Frozen components: {', '.join(frozen)}.")

    return TrainingProfile(
        strategy=strategy,
        parameters=parameters,
        trainable_components=trainable,
        frozen_components=frozen,
        observations=tuple(observations),
    )


def _extract_attributes(
    source: object | None,
    names: tuple[str, ...],
    prefix: str,
    output: dict[str, TrainingParameter],
) -> None:
    if source is None:
        return
    for name in names:
        value = _safe_getattr(source, name)
        normalized = _normalize_parameter(value)
        if normalized is not None:
            output[f"{prefix}.{name}"] = normalized


def _extract_peft_parameters(
    model: object | None,
    output: dict[str, TrainingParameter],
) -> bool:
    if model is None:
        return False
    raw_config = _safe_getattr(model, "peft_config")
    config: object | None = raw_config
    if isinstance(raw_config, Mapping):
        try:
            config = next(iter(raw_config.values()), None)
        except (RuntimeError, TypeError, ValueError):
            config = None
    if config is None:
        return False

    found = False
    for name in _PEFT_ARGUMENTS:
        normalized = _normalize_parameter(_safe_getattr(config, name))
        if normalized is None:
            continue
        output[f"peft.{name}"] = normalized
        found = True
    return found


def _component_states(model: object | None) -> dict[str, bool | None]:
    if model is None:
        return {}
    candidates = {
        "vision_tower": ("vision_tower", "vision_model"),
        "language_model": ("language_model", "text_model"),
        "mm_projector": (
            "mm_projector",
            "multi_modal_projector",
            "vision_projector",
            "projector",
        ),
    }
    states: dict[str, bool | None] = {}
    for label, names in candidates.items():
        component = _first_attribute(model, names)
        if component is not None:
            states[label] = _component_trainable_state(component)
    return states


def _first_attribute(value: object, names: tuple[str, ...]) -> object | None:
    for name in names:
        candidate = _safe_getattr(value, name)
        if candidate is not None:
            return candidate
    return None


def _component_trainable_state(component: object | None) -> bool | None:
    if component is None:
        return None
    parameters_method = _safe_getattr(component, "parameters")
    if not callable(parameters_method):
        return None
    try:
        parameters = parameters_method()
        iterator = iter(parameters)
    except Exception:
        return None

    seen = False
    try:
        for parameter in iterator:
            seen = True
            if _safe_getattr(parameter, "requires_grad") is True:
                return True
    except Exception:
        return None
    return False if seen else None


def _looks_like_vlm(
    model: object | None,
    config: object | None,
    components: Mapping[str, bool | None],
) -> bool:
    if "vision_tower" in components and (
        "language_model" in components or "mm_projector" in components
    ):
        return True
    model_type = _safe_getattr(config, "model_type")
    text = " ".join(
        (
            str(model_type or ""),
            getattr(type(model), "__name__", "") if model is not None else "",
        )
    ).lower()
    return any(token in text for token in _VLM_MODEL_TYPE_TOKENS)


def _strategy(
    *,
    is_vlm: bool,
    has_peft: bool,
    model_state: bool | None,
    components: Mapping[str, bool | None],
) -> TrainingStrategy:
    if is_vlm:
        if has_peft:
            return "vlm_adapter_finetune"
        if (
            components.get("mm_projector") is True
            and components.get("vision_tower") in {False, None}
            and components.get("language_model") in {False, None}
        ):
            return "vlm_projector_alignment"
        if any(state is True for state in components.values()) or model_state is True:
            return "vlm_full_finetune"
        return "unknown"
    if has_peft:
        return "adapter_finetune"
    if model_state is True:
        return "full_finetune"
    return "unknown"


def _normalize_parameter(value: object | None) -> TrainingParameter | None:
    if value is None:
        return None
    if isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        try:
            return tuple(str(item) for item in value)
        except (RuntimeError, TypeError, ValueError):
            return None
    if isinstance(value, set | frozenset):
        try:
            return tuple(sorted(str(item) for item in value))
        except (RuntimeError, TypeError, ValueError):
            return None
    return None


def _safe_getattr(value: object | None, name: str) -> object | None:
    if value is None:
        return None
    try:
        return getattr(value, name, None)
    except Exception:
        return None
