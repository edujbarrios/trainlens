from __future__ import annotations

from trainlens.llm.context import build_llm_notebook_context
from trainlens.pipeline import explain_namespace


class FakeParameter:
    def __init__(self, *, requires_grad: bool) -> None:
        self.requires_grad = requires_grad


class FakeComponent:
    def __init__(self, *, trainable: bool) -> None:
        self.trainable = trainable

    def parameters(self):
        return iter((FakeParameter(requires_grad=self.trainable),))


class FakeVLMConfig:
    model_type = "llava"
    mm_projector_type = "mlp2x_gelu"
    mm_vision_select_layer = -2


class FakeVLM:
    config = FakeVLMConfig()
    vision_tower = FakeComponent(trainable=False)
    language_model = FakeComponent(trainable=False)
    mm_projector = FakeComponent(trainable=True)


class FrozenVLM(FakeVLM):
    mm_projector = FakeComponent(trainable=False)


class FakeLoraConfig:
    r = 4
    lora_alpha = 16
    lora_dropout = 0.05
    target_modules = {"q_proj", "v_proj"}
    use_dora = True


class FakePeftVLM(FakeVLM):
    peft_config = {"default": FakeLoraConfig()}


class FakeTrainerState:
    log_history = [
        {"step": 1, "loss": 1.2},
        {"step": 2, "loss": 0.9, "eval_loss": 1.0},
    ]


class FakeTrainingArguments:
    learning_rate = 2e-5
    mm_projector_lr = 2e-4
    per_device_train_batch_size = 2
    gradient_accumulation_steps = 8
    max_length = 512


class FakeTrainer:
    state = FakeTrainerState()
    args = FakeTrainingArguments()
    model = FakePeftVLM()


FakeTrainer.__module__ = "transformers.trainer"


def test_pipeline_explains_vlm_adapter_training_configuration() -> None:
    result = explain_namespace({"trainer": FakeTrainer()})

    assert "Training strategy appears to be vlm adapter finetune." in result.summary
    assert "Learning rates: base=2e-05, projector=0.0002." in result.summary
    assert any("PEFT adapter: rank=4" in item for item in result.summary)

    titles = {signal.title for signal in result.signals}
    assert "Parameter-efficient VLM fine-tuning" in titles
    assert "Very small VLM adapter rank" in titles
    assert "Fixed multimodal sequence length" in titles
    assert "Multimodal learning-rate split" in titles

    actions = {item.action for item in result.recommendations}
    assert any("higher LoRA rank" in action for action in actions)
    assert any("preprocessed multimodal batch" in action for action in actions)


def test_llm_context_contains_normalized_vlm_training_profile() -> None:
    context = build_llm_notebook_context({"trainer": FakeTrainer()})

    assert "## Training Profile" in context.markdown
    assert "strategy: `vlm_adapter_finetune`" in context.markdown
    assert "`training.learning_rate`: 2e-05" in context.markdown
    assert "`training.mm_projector_lr`: 0.0002" in context.markdown
    assert "`training.max_length`: 512" in context.markdown
    assert "`peft.r`: 4" in context.markdown
    assert "`peft.target_modules`: ('q_proj', 'v_proj')" in context.markdown
    assert "frozen components: `vision_tower`, `language_model`" in context.markdown


def test_pipeline_detects_projector_only_alignment() -> None:
    trainer = FakeTrainer()
    trainer.model = FakeVLM()

    result = explain_namespace({"trainer": trainer})

    assert "Training strategy appears to be vlm projector alignment." in result.summary
    assert any(signal.title == "Projector-only VLM alignment" for signal in result.signals)
    assert any(
        "Validate projector alignment" in recommendation.action
        for recommendation in result.recommendations
    )


def test_pipeline_warns_when_all_detected_vlm_components_are_frozen() -> None:
    trainer = FakeTrainer()
    trainer.model = FrozenVLM()

    result = explain_namespace({"trainer": trainer})

    signal = next(
        item for item in result.signals if item.title == "All detected VLM components are frozen"
    )
    assert signal.severity == "warning"
    assert any(
        "Confirm the intended trainable VLM modules" in recommendation.action
        for recommendation in result.recommendations
    )
