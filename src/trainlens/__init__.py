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
from trainlens.leaderboard import LeaderboardRow, RunLeaderboard, leaderboard
from trainlens.llm.config import LLMConfig
from trainlens.llm.context import ContextPolicy
from trainlens.llm.openai_compatible import LLMRequestPreview, OpenAICompatibleProvider
from trainlens.llm.prompts import (
    PromptOptions,
    TrainLensPrompt,
    get_trainlens_prompt,
    show_trainlens_prompts,
)
from trainlens.llm.provider import LLMProvider
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
    preview_llm_request,
    preview_notebook_context,
)
from trainlens.pipeline import analyze
from trainlens.plotting import TrainingCurve, plot_training_curves, training_curves
from trainlens.project import Project, ProjectEntry
from trainlens.prompt_recipes import (
    controlled_experiment_prompt,
    improvement_plan_prompt,
    scientific_report_prompt,
    training_diagnosis_prompt,
)
from trainlens.repeats import (
    GroupComparison,
    MetricAggregate,
    ParameterEffect,
    RunGroup,
    aggregate_runs,
    compare_run_groups,
    parameter_effects,
)
from trainlens.runs import load_run, save_run, training_run_from_analysis
from trainlens.text import load_prompt, prompt_text
from trainlens.training_profile import TrainingProfile, inspect_training_profile

__version__ = "0.14.2"

__all__ = [
    "AlertDetector",
    "AnalysisConfig",
    "AnalysisResult",
    "ContextPolicy",
    "EvidenceRef",
    "ExperimentRun",
    "GroupComparison",
    "LLMConfig",
    "LLMProvider",
    "LLMRequestPreview",
    "LeaderboardRow",
    "LiveReport",
    "MetricAggregate",
    "MetricConstraint",
    "MetricRegistry",
    "MetricSpec",
    "MonitorConfig",
    "NextExperimentRecommendation",
    "ObjectiveSpec",
    "OpenAICompatibleProvider",
    "ParameterChange",
    "ParameterEffect",
    "Project",
    "ProjectEntry",
    "PromptOptions",
    "Recommendation",
    "RunGroup",
    "RunLeaderboard",
    "Signal",
    "SuccessCriterion",
    "TrainLensCallback",
    "TrainLensMonitor",
    "TrainLensPrompt",
    "TrainingAlert",
    "TrainingCurve",
    "TrainingObservation",
    "TrainingProfile",
    "TrainingRun",
    "TrajectoryComparison",
    "__version__",
    "aggregate_runs",
    "analyze",
    "build_improvement_ideas",
    "build_llm_report",
    "build_paper_report",
    "compare_run_groups",
    "compare_runs",
    "controlled_experiment_prompt",
    "default_metric_registry",
    "experiment_config",
    "get_trainlens_prompt",
    "improvement_plan_prompt",
    "inspect_training_profile",
    "keras_callback",
    "leaderboard",
    "lightning_callback",
    "load_ipython_extension",
    "load_prompt",
    "load_run",
    "parameter_effects",
    "pareto_front",
    "plot_training_curves",
    "preview_llm_request",
    "preview_notebook_context",
    "prompt_text",
    "register_metric",
    "render_next_experiment",
    "render_report",
    "render_run_comparison",
    "save_run",
    "scientific_report_prompt",
    "show_trainlens_prompts",
    "suggest_multiobjective_experiment",
    "suggest_next_experiment",
    "training_curves",
    "training_diagnosis_prompt",
    "training_run_from_analysis",
    "transformers_callback",
    "unload_ipython_extension",
    "unregister_metric",
    "write_report",
]
