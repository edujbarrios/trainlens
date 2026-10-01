"""TrainLens public API."""

from __future__ import annotations

from trainlens.analysis_config import AnalysisConfig
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
from trainlens.models.analysis import AnalysisResult, EvidenceRef, Recommendation, Signal
from trainlens.monitoring import MonitorConfig, TrainingAlert, TrainingObservation, TrainLensMonitor
from trainlens.notebook import (
    LiveReport,
    build_improvement_ideas,
    build_llm_report,
    build_paper_report,
    preview_notebook_context,
)
from trainlens.pipeline import analyze
from trainlens.training_profile import TrainingProfile, inspect_training_profile

__version__ = "0.10.1"

__all__ = [
    "AnalysisConfig",
    "AnalysisResult",
    "EvidenceRef",
    "ExperimentRun",
    "LiveReport",
    "MonitorConfig",
    "NextExperimentRecommendation",
    "PromptOptions",
    "Recommendation",
    "Signal",
    "SuccessCriterion",
    "TrainLensCallback",
    "TrainLensMonitor",
    "TrainLensPrompt",
    "TrainingAlert",
    "TrainingObservation",
    "TrainingProfile",
    "__version__",
    "analyze",
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
    "render_next_experiment",
    "render_report",
    "render_run_comparison",
    "show_trainlens_prompts",
    "suggest_next_experiment",
    "transformers_callback",
    "unload_ipython_extension",
    "write_report",
]
