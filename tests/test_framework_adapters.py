from __future__ import annotations

from trainlens.introspection import NotebookInspector
from trainlens.llm.context import build_llm_notebook_context
from trainlens.pipeline import explain_namespace


class ScalarTensor:
    def __init__(self, value: float) -> None:
        self.value = value

    def item(self) -> float:
        return self.value


class FakeKerasHistory:
    history = {
        "accuracy": [0.71, 0.84],
        "val_accuracy": [0.68, 0.79],
        "loss": [0.9, 0.42],
        "val_loss": [0.95, 0.51],
    }


FakeKerasHistory.__module__ = "keras.callbacks"


class FakeTransformerModel:
    pass


class FakeTrainerState:
    log_history = [
        {"step": 1, "loss": 1.2},
        {"step": 2, "loss": 0.8, "eval_loss": 0.9, "eval_accuracy": 0.76},
    ]


class FakeTrainingArguments:
    learning_rate = 0.00005
    num_train_epochs = 2


class FakeHuggingFaceTrainer:
    state = FakeTrainerState()
    args = FakeTrainingArguments()
    model = FakeTransformerModel()

    def evaluate(self) -> dict[str, float]:
        return {"eval_loss": 0.9}


FakeHuggingFaceTrainer.__module__ = "transformers.trainer"


class FakeLightningModule:
    pass


class FakeLightningTrainer:
    current_epoch = 3
    global_step = 120
    lightning_module = FakeLightningModule()
    callback_metrics = {
        "train_loss": ScalarTensor(0.33),
        "val_loss": ScalarTensor(0.47),
        "val_accuracy": ScalarTensor(0.88),
    }


FakeLightningTrainer.__module__ = "lightning.pytorch.trainer.trainer"


class FakeParameter:
    def __init__(self, size: int, *, requires_grad: bool) -> None:
        self.size = size
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return self.size


class FakeTorchModel:
    def parameters(self):
        return iter(
            (
                FakeParameter(80, requires_grad=True),
                FakeParameter(20, requires_grad=False),
            )
        )

    def state_dict(self) -> dict[str, object]:
        return {}

    def train(self) -> None:
        return None


class FakeAdamW:
    param_groups = [
        {
            "params": [object()],
            "lr": 0.0003,
            "weight_decay": 0.01,
            "betas": (0.9, 0.999),
            "custom_decay": 0.02,
        }
    ]
    defaults = {"eps": 1e-8, "amsgrad": False}


FakeAdamW.__module__ = "torch.optim.adamw"


class FakeCosineScheduler:
    last_epoch = 4

    def state_dict(self) -> dict[str, object]:
        return {"T_max": 10, "eta_min": 1e-6, "base_lrs": [0.0003]}

    def get_last_lr(self) -> list[float]:
        return [0.00015]


FakeCosineScheduler.__module__ = "torch.optim.lr_scheduler"


class FakeDataset:
    def __len__(self) -> int:
        return 128


class FakeDataLoader:
    dataset = FakeDataset()
    batch_size = 16
    num_workers = 4
    drop_last = True
    pin_memory = True
    persistent_workers = True
    prefetch_factor = 2
    timeout = 0


FakeDataLoader.__module__ = "torch.utils.data.dataloader"


def test_inspector_extracts_keras_history_artifact() -> None:
    snapshot = NotebookInspector().snapshot({"history": FakeKerasHistory()})

    assert len(snapshot.framework_artifacts) == 1
    artifact = snapshot.framework_artifacts[0]
    assert artifact.framework == "keras"
    assert artifact.history["val_accuracy"] == (0.68, 0.79)


def test_pipeline_uses_keras_history_metrics() -> None:
    result = explain_namespace({"history": FakeKerasHistory()})

    assert result.framework == "keras"
    assert result.metrics["train_accuracy"] == 0.84
    assert result.metrics["validation_loss"] == 0.51
    assert "Adapted keras metrics from `history`." in result.summary


def test_pipeline_uses_huggingface_trainer_logs_and_model() -> None:
    result = explain_namespace({"trainer": FakeHuggingFaceTrainer()})

    assert result.framework == "huggingface"
    assert result.model_name == "FakeTransformerModel"
    assert result.metrics["train_loss"] == 0.8
    assert result.metrics["validation_accuracy"] == 0.76


def test_pipeline_uses_lightning_callback_metrics() -> None:
    result = explain_namespace({"trainer": FakeLightningTrainer()})

    assert result.framework == "lightning"
    assert result.model_name == "FakeLightningModule"
    assert result.metrics["train_loss"] == 0.33
    assert result.metrics["validation_loss"] == 0.47


def test_inspector_extracts_plain_pytorch_training_parameters() -> None:
    snapshot = NotebookInspector().snapshot(
        {
            "model": FakeTorchModel(),
            "optimizer": FakeAdamW(),
            "scheduler": FakeCosineScheduler(),
            "train_loader": FakeDataLoader(),
        }
    )

    by_name = {artifact.variable_name: artifact for artifact in snapshot.framework_artifacts}
    assert by_name["model"].training_parameters == {
        "total_parameters": 100,
        "trainable_parameters": 80,
        "trainable_ratio": 0.8,
    }
    assert by_name["optimizer"].training_parameters["group_0.lr"] == 0.0003
    assert by_name["optimizer"].training_parameters["group_0.betas"] == (0.9, 0.999)
    assert by_name["optimizer"].training_parameters["group_0.custom_decay"] == 0.02
    assert "params" not in " ".join(by_name["optimizer"].training_parameters)
    assert by_name["scheduler"].training_parameters["last_lr"] == (0.00015,)
    assert by_name["scheduler"].training_parameters["T_max"] == 10
    assert by_name["scheduler"].training_parameters["base_lrs"] == (0.0003,)
    assert by_name["train_loader"].training_parameters["dataset_size"] == 128
    assert by_name["train_loader"].training_parameters["batch_size"] == 16
    assert by_name["train_loader"].training_parameters["prefetch_factor"] == 2


def test_llm_context_includes_plain_pytorch_training_parameters() -> None:
    context = build_llm_notebook_context(
        {
            "model": FakeTorchModel(),
            "optimizer": FakeAdamW(),
            "scheduler": FakeCosineScheduler(),
            "train_loader": FakeDataLoader(),
        }
    )

    assert "## Training Parameters" in context.markdown
    assert "`trainable_parameters`: 80" in context.markdown
    assert "`group_0.lr`: 0.0003" in context.markdown
    assert "`last_lr`: (0.00015,)" in context.markdown
    assert "`batch_size`: 16" in context.markdown
