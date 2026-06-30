"""IPython extension entry points."""

# mypy: disable-error-code="no-untyped-call"

from __future__ import annotations

from IPython.core.interactiveshell import InteractiveShell

from trainlens.magic.commands import TrainLensMagics


def load_ipython_extension(ipython: InteractiveShell) -> None:
    ipython.register_magics(TrainLensMagics)


def unload_ipython_extension(ipython: InteractiveShell) -> None:
    ipython.magics_manager.magics["line"].pop("explain_training", None)
    ipython.magics_manager.magics["line"].pop("why_bad_model", None)
    ipython.magics_manager.magics["line"].pop("training_summary", None)
    ipython.magics_manager.magics["line"].pop("compare_runs", None)
