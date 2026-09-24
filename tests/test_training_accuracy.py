from trainlens.pipeline import explain_namespace


class FakeLightningTrainer:
    current_epoch = 0
    global_step = 1
    callback_metrics = {"train_accuracy": 0.91}


FakeLightningTrainer.__module__ = "lightning.pytorch.trainer.trainer"


def test_preserves_single_point_training_accuracy() -> None:
    result = explain_namespace({"trainer": FakeLightningTrainer()})

    assert result.metrics["train_accuracy"] == 0.91
    assert all("Training accuracy changed" not in item for item in result.summary)
