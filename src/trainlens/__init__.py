"""TrainLens public API."""

from __future__ import annotations

from trainlens.export import render_report, write_report
from trainlens.magic.extension import load_ipython_extension, unload_ipython_extension
from trainlens.notebook import (
    LiveReport,
    build_improvement_ideas,
    build_llm_report,
    build_paper_report,
)

__version__ = "0.4.0"

__all__ = [
    "LiveReport",
    "__version__",
    "build_improvement_ideas",
    "build_llm_report",
    "build_paper_report",
    "load_ipython_extension",
    "render_report",
    "unload_ipython_extension",
    "write_report",
]
