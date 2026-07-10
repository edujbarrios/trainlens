from __future__ import annotations

from trainlens.introspection import NotebookInspector
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
