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
from trainlens.models.comparison import ParameterChange, TrajectoryComparison
from trainlens.models.run import TrainingRun
from trainlens.monitoring import (
    AlertDetector,
    MonitorConfig,
    TrainingAlert,
    TrainingObservation,
    TrainLensMonitor,
)
from trainlens.notebook import (
    LiveReport,
    build_improvement_ideas,
    build_llm_report,
    build_paper_report,
    preview_notebook_context,
)
from trainlens.pipeline import analyze
from trainlens.runs import load_run, save_run, training_run_from_analysis
from trainlens.training_profile import TrainingProfile, inspect_training_profile

__version__ = "0.10.1"

__all__ = [
    "AlertDetector",
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
    "ParameterChange",
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
    "TrainingRun",
    "TrajectoryComparison",
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
    "load_run",
    "pareto_front",
    "preview_notebook_context",
    "register_metric",
    "render_next_experiment",
    "render_report",
    "render_run_comparison",
    "save_run",
    "show_trainlens_prompts",
    "suggest_multiobjective_experiment",
    "suggest_next_experiment",
    "training_run_from_analysis",
    "transformers_callback",
    "unload_ipython_extension",
    "unregister_metric",
    "write_report",
]
