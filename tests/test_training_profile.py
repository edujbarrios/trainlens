from __future__ import annotations

import pytest

from trainlens import inspect_training_profile


class FakeParameter:
    def __init__(self, *, requires_grad: bool) -> None:
        self.requires_grad = requires_grad


class SizedParameter(FakeParameter):
    def __init__(self, size: int, *, requires_grad: bool) -> None:
        super().__init__(requires_grad=requires_grad)
        self.size = size

    def numel(self) -> int:
        return self.size


class FakeComponent:
    def __init__(self, *, trainable: bool) -> None:
        self.trainable = trainable

    def parameters(self):
        return iter((FakeParameter(requires_grad=self.trainable),))


class FakeVLMConfig:
    model_type = "llava"
    mm_projector_type = "mlp2x_gelu"
    mm_vision_select_layer = -2
    image_aspect_ratio = "pad"


class FakeVLM:
    config = FakeVLMConfig()
    vision_tower = FakeComponent(trainable=False)
    language_model = FakeComponent(trainable=False)
    mm_projector = FakeComponent(trainable=True)

    def parameters(self):
        return iter(
            (
                FakeParameter(requires_grad=False),
                FakeParameter(requires_grad=True),
            )
        )


class FakeTrainingArguments:
    learning_rate = 2e-5
    weight_decay = 0.01
    per_device_train_batch_size = 2
    gradient_accumulation_steps = 8
    gradient_checkpointing = True
    bf16 = True
    mm_projector_lr = 2e-4


class FakeTrainer:
    model = FakeVLM()
    args = FakeTrainingArguments()


class FakeLoraConfig:
    r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    target_modules = {"q_proj", "v_proj"}
    bias = "none"
    use_rslora = False
    use_dora = True


class SmallLoraConfig(FakeLoraConfig):
    r = 4


class LargeLoraConfig(FakeLoraConfig):
    r = 32


class FakePeftVLM(FakeVLM):
    peft_config = {"default": FakeLoraConfig()}


class MultiAdapterModel(FakeVLM):
    peft_config = {
        "small": SmallLoraConfig(),
        "large": LargeLoraConfig(),
    }
    active_adapter = "large"


class PartialModel:
    def parameters(self):
        return iter(
            (
                SizedParameter(100, requires_grad=True),
                SizedParameter(900, requires_grad=False),
            )
        )


class FullModel:
    def parameters(self):
        return iter(
            (
                SizedParameter(400, requires_grad=True),
                SizedParameter(600, requires_grad=True),
            )
        )


class QuantizedModel(PartialModel):
    is_loaded_in_4bit = True


class BrokenComponent:
    @property
    def parameters(self):
        raise RuntimeError("unavailable")


class BrokenVLM(FakeVLM):
    vision_tower = BrokenComponent()


def test_detects_projector_only_vlm_alignment() -> None:
    profile = inspect_training_profile(FakeVLM())

    assert profile.strategy == "vlm_projector_alignment"
    assert profile.trainable_components == ("mm_projector",)
    assert profile.frozen_components == ("vision_tower", "language_model")
    assert profile.parameters["vlm.model_type"] == "llava"
    assert profile.parameters["vlm.mm_projector_type"] == "mlp2x_gelu"


def test_extracts_huggingface_training_arguments() -> None:
    profile = inspect_training_profile(trainer=FakeTrainer())

    assert profile.parameters["training.learning_rate"] == 2e-5
    assert profile.parameters["training.gradient_accumulation_steps"] == 8
    assert profile.parameters["training.gradient_checkpointing"] is True
    assert profile.parameters["training.mm_projector_lr"] == 2e-4


def test_extracts_peft_lora_configuration() -> None:
    trainer = FakeTrainer()
    trainer.model = FakePeftVLM()

    profile = inspect_training_profile(trainer=trainer)

    assert profile.strategy == "vlm_adapter_finetune"
    assert profile.parameters["peft.r"] == 16
    assert profile.parameters["peft.lora_alpha"] == 32
    assert profile.parameters["peft.target_modules"] == ("q_proj", "v_proj")
    assert profile.parameters["peft.use_dora"] is True
    assert profile.adapters == ("default",)


def test_reads_common_notebook_aliases() -> None:
    profile = inspect_training_profile(
        FakeVLM(),
        namespace={
            "lora_rank": 8,
            "lora_alpha": 16,
            "vision_tower_lr": 1e-5,
            "tune_mm_mlp_adapter": True,
        },
    )

    assert profile.parameters["peft.r"] == 8
    assert profile.parameters["peft.lora_alpha"] == 16
    assert profile.parameters["training.vision_tower_lr"] == 1e-5
    assert profile.parameters["vlm.tune_mm_mlp_adapter"] is True


def test_broken_component_introspection_is_best_effort() -> None:
    profile = inspect_training_profile(BrokenVLM())

    assert profile.strategy == "vlm_projector_alignment"
    assert "vision_tower" not in profile.frozen_components
    assert profile.parameters["vlm.model_type"] == "llava"


def test_detects_partial_finetuning_and_parameter_fraction() -> None:
    profile = inspect_training_profile(PartialModel())

    assert profile.strategy == "partial_finetune"
    assert profile.trainable_parameters == 100
    assert profile.total_parameters == 1000
    assert profile.trainable_fraction == pytest.approx(0.1)
    assert any("10.00%" in item for item in profile.observations)


def test_detects_full_finetuning_from_parameter_counts() -> None:
    profile = inspect_training_profile(FullModel())

    assert profile.strategy == "full_finetune"
    assert profile.trainable_parameters == 1000
    assert profile.total_parameters == 1000
    assert profile.trainable_fraction == pytest.approx(1.0)


def test_tracks_multiple_peft_adapters_and_active_configuration() -> None:
    profile = inspect_training_profile(MultiAdapterModel())

    assert profile.strategy == "vlm_adapter_finetune"
    assert profile.adapters == ("small", "large")
    assert profile.active_adapters == ("large",)
    assert profile.parameters["peft.r"] == 32
    assert profile.parameters["peft.adapters"] == ("small", "large")
    assert profile.parameters["peft.active_adapters"] == ("large",)


def test_detects_quantized_training_configuration() -> None:
    profile = inspect_training_profile(QuantizedModel())

    assert profile.quantization == "4bit"
    assert profile.parameters["training.quantization"] == "4bit"
