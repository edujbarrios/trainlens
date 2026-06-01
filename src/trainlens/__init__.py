"""TrainLens public API."""

from __future__ import annotations

from trainlens.magic.extension import load_ipython_extension, unload_ipython_extension
from trainlens.notebook import LiveReport, build_live_report, display_live_report
from trainlens.pipeline import explain_namespace

__all__ = [
    "LiveReport",
    "build_live_report",
    "display_live_report",
    "explain_namespace",
    "load_ipython_extension",
    "unload_ipython_extension",
]
