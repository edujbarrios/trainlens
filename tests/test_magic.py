from trainlens.magic.commands import TrainLensMagics


class DemoShell:
    user_ns = {
        "history": {"train_loss": [2.0, 1.5], "eval_loss": [2.1, 1.8]},
        "training_trace": [{"step": 1, "loss": 2.0}, {"step": 2, "eval_loss": 1.8}],
    }


def test_training_dashboard_magic_captures_run():
    magics = TrainLensMagics(DemoShell())

    magics.training_dashboard("")

    assert magics.store.latest() is not None
