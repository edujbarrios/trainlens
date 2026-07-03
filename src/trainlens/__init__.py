"""TrainLens public API."""

from __future__ import annotations

from trainlens.magic.extension import load_ipython_extension, unload_ipython_extension
from trainlens.notebook import (
    LiveReport,
    build_llm_report,
)

__version__ = "0.1.0"

__all__ = [
    "LiveReport",
    "__version__",
    "build_llm_report",
    "load_ipython_extension",
    "unload_ipython_extension",
]
