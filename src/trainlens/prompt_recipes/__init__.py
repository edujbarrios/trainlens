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

__all__ = [
    "controlled_experiment_prompt",
    "improvement_plan_prompt",
    "overfitting_review_prompt",
    "prompt_options",
    "scientific_report_prompt",
    "training_diagnosis_prompt",
]
