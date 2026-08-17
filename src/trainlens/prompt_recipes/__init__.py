"""Reusable prompt recipes for TrainLens report APIs."""

from trainlens.prompt_recipes.base import prompt_options
from trainlens.prompt_recipes.diagnostics import (
    overfitting_review_prompt,
    training_diagnosis_prompt,
)
from trainlens.prompt_recipes.experiments import (
    controlled_experiment_prompt,
    improvement_plan_prompt,
)
from trainlens.prompt_recipes.reports import scientific_report_prompt
from trainlens.prompt_recipes.scientific_designs import (
    ablation_study_prompt,
    hypothesis_test_prompt,
    reproducibility_audit_prompt,
    sensitivity_analysis_prompt,
)

__all__ = [
    "ablation_study_prompt",
    "controlled_experiment_prompt",
    "improvement_plan_prompt",
    "hypothesis_test_prompt",
    "overfitting_review_prompt",
    "prompt_options",
    "reproducibility_audit_prompt",
    "scientific_report_prompt",
    "sensitivity_analysis_prompt",
    "training_diagnosis_prompt",
]
