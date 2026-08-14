"""TrainLens public API."""

from __future__ import annotations

from trainlens.comparison import compare_runs, render_run_comparison
from trainlens.export import render_report, write_report
from trainlens.llm.prompts import TrainLensPrompt, get_trainlens_prompt, show_trainlens_prompts
from trainlens.magic.extension import load_ipython_extension, unload_ipython_extension
from trainlens.notebook import (
    LiveReport,
    build_improvement_ideas,
    build_llm_report,
    build_paper_report,
)

__version__ = "0.7.0"

__all__ = [
    "LiveReport",
    "TrainLensPrompt",
    "__version__",
    "build_improvement_ideas",
    "build_llm_report",
    "build_paper_report",
    "compare_runs",
    "get_trainlens_prompt",
    "load_ipython_extension",
    "render_report",
    "render_run_comparison",
    "show_trainlens_prompts",
    "unload_ipython_extension",
    "write_report",
]
