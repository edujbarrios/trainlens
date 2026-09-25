"""IPython extension entry points."""

# mypy: disable-error-code="no-untyped-call"

from __future__ import annotations

from IPython.core.interactiveshell import InteractiveShell

from trainlens.magic.commands import TrainLensMagics

_MAGIC_ATTRIBUTE = "_trainlens_magics"
_MAGIC_NAMES = ("explain_training", "suggest_improvements", "compare_runs")


def load_ipython_extension(ipython: InteractiveShell) -> None:
    """Register TrainLens magics, reusing notebook-local state across reloads."""

    magics = getattr(ipython, _MAGIC_ATTRIBUTE, None)
    if not isinstance(magics, TrainLensMagics):
        magics = TrainLensMagics(ipython)
        setattr(ipython, _MAGIC_ATTRIBUTE, magics)
    ipython.register_magics(magics)


def unload_ipython_extension(ipython: InteractiveShell) -> None:
    """Unregister TrainLens line magics without discarding captured run state."""

    line_magics = ipython.magics_manager.magics["line"]
    for name in _MAGIC_NAMES:
        line_magics.pop(name, None)
