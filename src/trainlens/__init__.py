"""TrainLens public API."""

from __future__ import annotations

from trainlens.callbacks import TrainLensCallback
from trainlens.comparison import compare_runs, render_run_comparison
from trainlens.experiments import (
    ExperimentRun,
    NextExperimentRecommendation,
    SuccessCriterion,
    experiment_config,
    render_next_experiment,
    suggest_next_experiment,
)
from trainlens.export import render_report, write_report
from trainlens.framework_callbacks import keras_callback, lightning_callback, transformers_callback
from trainlens.llm.prompts import (
    PromptOptions,
    TrainLensPrompt,
    get_trainlens_prompt,
    show_trainlens_prompts,
)
from trainlens.magic.extension import load_ipython_extension, unload_ipython_extension
from trainlens.monitoring import MonitorConfig, TrainingAlert, TrainingObservation, TrainLensMonitor
from trainlens.notebook import (
    LiveReport,
    build_improvement_ideas,
    build_llm_report,
    build_paper_report,
    preview_notebook_context,
)
from trainlens.training_profile import TrainingProfile, inspect_training_profile

__version__ = "0.10.0"

__all__ = [
    "LiveReport",
    "ExperimentRun",
    "MonitorConfig",
    "NextExperimentRecommendation",
    "PromptOptions",
    "TrainLensPrompt",
    "TrainingAlert",
    "TrainingObservation",
    "TrainingProfile",
    "SuccessCriterion",
    "TrainLensMonitor",
    "TrainLensCallback",
    "__version__",
    "build_improvement_ideas",
    "build_llm_report",
    "build_paper_report",
    "compare_runs",
    "experiment_config",
    "get_trainlens_prompt",
    "inspect_training_profile",
    "keras_callback",
    "lightning_callback",
    "load_ipython_extension",
    "preview_notebook_context",
    "render_report",
    "render_next_experiment",
    "render_run_comparison",
    "show_trainlens_prompts",
    "suggest_next_experiment",
    "transformers_callback",
    "unload_ipython_extension",
    "write_report",
]
