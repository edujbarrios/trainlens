from trainlens.magic.commands import TrainLensMagics


class DemoShell:
    user_ns = {
        "history": {"train_loss": [2.0, 1.5], "eval_loss": [2.1, 1.8]},
        "training_trace": [{"step": 1, "loss": 2.0}, {"step": 2, "eval_loss": 1.8}],
    }


def test_explain_training_magic_captures_run():
    magics = TrainLensMagics(DemoShell())

    magics.explain_training("")

    assert magics.store.latest() is not None


def test_training_atlas_magic_captures_run():
    magics = TrainLensMagics(DemoShell())

    magics.training_atlas("")

    assert magics.store.latest() is not None
