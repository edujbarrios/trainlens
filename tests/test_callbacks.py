from types import SimpleNamespace

import pytest

from trainlens import MonitorConfig, TrainLensCallback, TrainLensMonitor


def test_keras_hook_streams_logs_and_can_request_stop_on_critical_alert():
    callback = TrainLensCallback(stop_on_anomaly=True)
    model = SimpleNamespace(stop_training=False)
    callback.set_model(model)

    callback.on_epoch_end(4, {"loss": float("nan")})

    assert callback.alerts[0].code == "non_finite_metric"
    assert callback.stop_requested is True
    assert model.stop_training is True


def test_transformers_hook_uses_global_step_and_updates_control():
    callback = TrainLensCallback(stop_on_anomaly=True)
    state = SimpleNamespace(global_step=25)
    control = SimpleNamespace(should_training_stop=False)

    returned = callback.on_log(state=state, control=control, logs={"loss": float("inf")})

    assert returned is control
    assert control.should_training_stop is True
    assert callback.monitor.observations[0].step == 25


def test_lightning_hook_reads_callback_metrics_without_framework_dependency():
    callback = TrainLensCallback(
        TrainLensMonitor(MonitorConfig(patience=2, min_delta=0.01))
    )
    trainer = SimpleNamespace(current_epoch=3, callback_metrics={"train_loss": 0.4})

    callback.on_train_epoch_end(trainer)

    assert callback.monitor.observations[0].metrics == {"train_loss": 0.4}


def test_callback_triggers_configured_periodic_explanation_handler():
    observations = []
    callback = TrainLensCallback(explain_every=2, on_explain=observations.append)

    callback.observe(1, {"loss": 1.0})
    callback.observe(2, {"loss": 0.8})

    assert [observation.step for observation in observations] == [2]


def test_callback_ignores_metadata_and_normalizes_tensor_like_scalars():
    class Scalar:
        def item(self):
            return 0.75

    callback = TrainLensCallback()
    callback.observe(1, {"loss": Scalar(), "phase": "train", "ready": True})

    assert callback.monitor.observations[0].metrics == {"loss": 0.75}


def test_callback_ignores_tensor_like_values_that_are_not_scalars():
    class NonScalar:
        def item(self):
            raise ValueError("only one element tensors can be converted to scalars")

    callback = TrainLensCallback()

    callback.observe(1, {"loss": 0.75, "confusion_matrix": NonScalar()})

    assert callback.monitor.observations[0].metrics == {"loss": 0.75}


@pytest.mark.parametrize("explain_every", [True, 1.5, "2"])
def test_callback_rejects_non_integer_explanation_intervals(explain_every):
    with pytest.raises(TypeError, match="explain_every must be an integer or None"):
        TrainLensCallback(explain_every=explain_every, on_explain=lambda observation: None)


def test_callback_ignores_tensor_like_boolean_metrics():
    class BooleanScalar:
        def item(self):
            return True

    callback = TrainLensCallback()
    callback.observe(1, {"ready": BooleanScalar(), "loss": 0.75})

    assert callback.monitor.observations[0].metrics == {"loss": 0.75}


def test_callback_explanation_interval_counts_completed_observations_from_epoch_zero():
    explained = []
    callback = TrainLensCallback(explain_every=5, on_explain=explained.append)

    for epoch in range(6):
        callback.on_epoch_end(epoch, {"loss": 1.0})

    assert [observation.step for observation in explained] == [4]


def test_callback_implements_keras_lifecycle_protocol():
    callback = TrainLensCallback()
    callback.set_params({"epochs": 3})

    assert callback.params == {"epochs": 3}
    for name in (
        "on_train_begin",
        "on_train_end",
        "on_epoch_begin",
        "on_train_batch_begin",
        "on_train_batch_end",
        "on_test_begin",
        "on_test_end",
        "on_predict_begin",
        "on_predict_end",
    ):
        assert callable(getattr(callback, name))


def test_callback_implements_transformers_lifecycle_protocol_and_returns_control():
    callback = TrainLensCallback()
    control = SimpleNamespace(should_training_stop=False)

    for name in (
        "on_init_end",
        "on_train_begin",
        "on_train_end",
        "on_epoch_begin",
        "on_evaluate",
        "on_save",
        "on_prediction_step",
        "on_step_begin",
        "on_step_end",
        "on_substep_end",
    ):
        assert getattr(callback, name)(object(), object(), control) is control

    assert callback.on_epoch_end(object(), object(), control) is control
