"""Optional framework-native adapters for :class:`TrainLensCallback`."""

# mypy: disable-error-code="misc,valid-type,no-any-return"

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from typing import Any

from trainlens.callbacks import TrainLensCallback


def keras_callback(
    callback: TrainLensCallback | None = None,
    **callback_options: Any,
) -> Any:
    """Return a Keras-native callback that delegates to TrainLens.

    Keras remains optional: the import only occurs when this factory is called.
    """

    delegate = _delegate(callback, callback_options)
    try:
        callback_base = import_module("keras.callbacks").Callback
    except (AttributeError, ImportError) as exc:
        raise RuntimeError(
            "Keras is required for keras_callback(); install a supported Keras version first."
        ) from exc

    class KerasTrainLensCallback(callback_base):  # type: ignore[misc,valid-type]
        def __init__(self) -> None:
            super().__init__()
            self.trainlens = delegate

        def on_epoch_end(
            self,
            epoch: int,
            logs: Mapping[str, Any] | None = None,
        ) -> None:
            model = getattr(self, "model", None)
            if model is not None:
                self.trainlens.set_model(model)
            self.trainlens.on_epoch_end(epoch, logs)

    return KerasTrainLensCallback()


def transformers_callback(
    callback: TrainLensCallback | None = None,
    **callback_options: Any,
) -> Any:
    """Return a Transformers ``TrainerCallback`` that delegates logging to TrainLens."""

    delegate = _delegate(callback, callback_options)
    try:
        callback_base = import_module("transformers").TrainerCallback
    except (AttributeError, ImportError) as exc:
        raise RuntimeError(
            "Transformers is required for transformers_callback(); install transformers first."
        ) from exc

    class TransformersTrainLensCallback(callback_base):  # type: ignore[misc,valid-type]
        def __init__(self) -> None:
            super().__init__()
            self.trainlens = delegate

        def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: Mapping[str, Any] | None = None,
            **kwargs: Any,
        ) -> Any:
            return self.trainlens.on_log(
                args=args,
                state=state,
                control=control,
                logs=logs,
                **kwargs,
            )

    return TransformersTrainLensCallback()


def lightning_callback(
    callback: TrainLensCallback | None = None,
    **callback_options: Any,
) -> Any:
    """Return a Lightning-native callback that delegates epoch metrics to TrainLens."""

    delegate = _delegate(callback, callback_options)
    callback_base = _lightning_callback_base()

    class LightningTrainLensCallback(callback_base):  # type: ignore[misc,valid-type]
        def __init__(self) -> None:
            super().__init__()
            self.trainlens = delegate

        def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
            self.trainlens.on_train_epoch_end(trainer, pl_module)

    return LightningTrainLensCallback()


def _lightning_callback_base() -> Any:
    errors: list[BaseException] = []
    for module_name in ("lightning.pytorch.callbacks", "pytorch_lightning.callbacks"):
        try:
            return import_module(module_name).Callback
        except (AttributeError, ImportError) as exc:
            errors.append(exc)
    raise RuntimeError(
        "Lightning is required for lightning_callback(); install lightning or pytorch-lightning first."
    ) from errors[-1]


def _delegate(
    callback: TrainLensCallback | None,
    callback_options: Mapping[str, Any],
) -> TrainLensCallback:
    if callback is not None and callback_options:
        raise ValueError("pass either an existing callback or callback options, not both")
    return callback or TrainLensCallback(**dict(callback_options))
