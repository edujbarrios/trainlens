from __future__ import annotations

from types import SimpleNamespace

import pytest

from trainlens import TrainLensCallback, keras_callback, lightning_callback, transformers_callback


class FakeKerasCallback:
    def __init__(self) -> None:
        self.model = None
        self.params = {}

    def set_model(self, model) -> None:
        self.model = model

    def set_params(self, params) -> None:
        self.params = params

    def on_batch_end(self, batch, logs=None) -> None:
        return None

    def on_train_batch_end(self, batch, logs=None) -> None:
        return None

    def on_test_batch_end(self, batch, logs=None) -> None:
        return None

    def on_predict_batch_end(self, batch, logs=None) -> None:
        return None


class FakeTrainerCallback:
    def on_init_end(self, args, state, control, **kwargs):
        return control

    def on_train_begin(self, args, state, control, **kwargs):
        return control

    def on_epoch_begin(self, args, state, control, **kwargs):
        return control

    def on_evaluate(self, args, state, control, **kwargs):
        return control


class FakeLightningCallback:
    @property
    def state_key(self) -> str:
        return self.__class__.__qualname__

    def setup(self, trainer, pl_module, stage) -> None:
        return None


def _fake_import_module(name: str):
    if name == "keras.callbacks":
        return SimpleNamespace(Callback=FakeKerasCallback)
    if name == "transformers":
        return SimpleNamespace(TrainerCallback=FakeTrainerCallback)
    if name == "lightning.pytorch.callbacks":
        return SimpleNamespace(Callback=FakeLightningCallback)
    raise ImportError(name)


def test_keras_factory_inherits_callback_contract_and_streams_epoch_metrics(monkeypatch):
    monkeypatch.setattr("trainlens.framework_callbacks.import_module", _fake_import_module)
    core = TrainLensCallback(stop_on_anomaly=True)
    native = keras_callback(core)
    model = SimpleNamespace(stop_training=False)

    native.set_model(model)
    native.set_params({"epochs": 3})
    native.on_batch_end(0, {})
    native.on_train_batch_end(0, {})
    native.on_test_batch_end(0, {})
    native.on_predict_batch_end(0, {})
    native.on_epoch_end(1, {"loss": float("nan")})

    assert native.trainlens is core
    assert core.monitor.observations[0].step == 1
    assert core.alerts[0].code == "non_finite_metric"
    assert model.stop_training is True


def test_transformers_factory_inherits_full_event_surface_and_delegates_on_log(monkeypatch):
    monkeypatch.setattr("trainlens.framework_callbacks.import_module", _fake_import_module)
    native = transformers_callback(stop_on_anomaly=True)
    args = SimpleNamespace()
    state = SimpleNamespace(global_step=42)
    control = SimpleNamespace(should_training_stop=False)

    assert native.on_init_end(args, state, control) is control
    assert native.on_train_begin(args, state, control) is control
    assert native.on_epoch_begin(args, state, control) is control
    assert native.on_evaluate(args, state, control) is control
    returned = native.on_log(args, state, control, logs={"loss": float("inf")})

    assert returned is control
    assert native.trainlens.monitor.observations[0].step == 42
    assert control.should_training_stop is True


def test_lightning_factory_inherits_callback_metadata_and_delegates_epoch_metrics(monkeypatch):
    monkeypatch.setattr("trainlens.framework_callbacks.import_module", _fake_import_module)
    native = lightning_callback(stop_on_anomaly=True)
    trainer = SimpleNamespace(
        current_epoch=7,
        callback_metrics={"loss": float("nan")},
        should_stop=False,
    )

    assert native.state_key
    native.setup(trainer, object(), "fit")
    native.on_train_epoch_end(trainer, object())

    assert native.trainlens.monitor.observations[0].step == 7
    assert trainer.should_stop is True


def test_callback_factories_keep_framework_dependencies_optional(monkeypatch):
    def missing(_name: str):
        raise ImportError("not installed")

    monkeypatch.setattr("trainlens.framework_callbacks.import_module", missing)

    with pytest.raises(RuntimeError, match="Keras is required"):
        keras_callback()
    with pytest.raises(RuntimeError, match="Transformers is required"):
        transformers_callback()
    with pytest.raises(RuntimeError, match="Lightning is required"):
        lightning_callback()


def test_callback_factory_rejects_delegate_and_constructor_options_together(monkeypatch):
    monkeypatch.setattr("trainlens.framework_callbacks.import_module", _fake_import_module)

    with pytest.raises(ValueError, match="either an existing callback or callback options"):
        keras_callback(TrainLensCallback(), stop_on_anomaly=True)
