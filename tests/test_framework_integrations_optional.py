"""Optional smoke tests against actual supported training framework classes."""

from __future__ import annotations

import pytest

from trainlens import keras_callback, lightning_callback, transformers_callback


def test_native_keras_callback_inheritance() -> None:
    keras = pytest.importorskip("keras")
    callback = keras_callback()
    assert isinstance(callback, keras.callbacks.Callback)


def test_native_transformers_callback_inheritance() -> None:
    transformers = pytest.importorskip("transformers")
    callback = transformers_callback()
    assert isinstance(callback, transformers.TrainerCallback)


def test_native_lightning_callback_inheritance() -> None:
    callbacks = pytest.importorskip("lightning.pytorch.callbacks")
    callback = lightning_callback()
    assert isinstance(callback, callbacks.Callback)
