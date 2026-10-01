"""TrainLens public API."""

from __future__ import annotations

from trainlens.analysis_config import AnalysisConfig
from trainlens.callbacks import TrainLensCallback
from trainlens.comparison import compare_runs, render_run_comparison
from trainlens.experiments import (
    ExperimentRun,
    MetricConstraint,
    NextExperimentRecommendation,
    ObjectiveSpec,
    SuccessCriterion,
    experiment_config,
    pareto_front,
    render_next_experiment,
    suggest_multiobjective_experiment,
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
from trainlens.metric_semantics import (
    MetricRegistry,
    MetricSpec,
    default_metric_registry,
    register_metric,
    unregister_metric,
)
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
    "MetricConstraint",
    "MetricRegistry",
    "MetricSpec",
    "MonitorConfig",
    "NextExperimentRecommendation",
    "ObjectiveSpec",
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
    "default_metric_registry",
    "experiment_config",
    "get_trainlens_prompt",
    "inspect_training_profile",
    "keras_callback",
    "lightning_callback",
    "load_ipython_extension",
    "pareto_front",
    "preview_notebook_context",
    "register_metric",
    "render_next_experiment",
    "render_report",
    "render_run_comparison",
    "show_trainlens_prompts",
    "suggest_multiobjective_experiment",
    "suggest_next_experiment",
    "transformers_callback",
    "unload_ipython_extension",
    "unregister_metric",
    "write_report",
]
