from trainlens import TrainLensCallback


def test_callback_explanation_runs_once_for_repeated_step_updates() -> None:
    explained_steps: list[int] = []
    callback = TrainLensCallback(
        explain_every=2,
        on_explain=lambda observation: explained_steps.append(observation.step),
    )

    callback.observe(2, {"loss": 1.0})
    callback.observe(2, {"accuracy": 0.5})

    assert explained_steps == [2]
    assert callback.monitor.observations[-1].metrics == {"loss": 1.0, "accuracy": 0.5}
