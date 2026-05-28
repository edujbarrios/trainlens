"""TrainLens public API."""

from __future__ import annotations

from trainlens.magic.extension import load_ipython_extension, unload_ipython_extension
from trainlens.pipeline import explain_namespace

__all__ = ["explain_namespace", "load_ipython_extension", "unload_ipython_extension"]
