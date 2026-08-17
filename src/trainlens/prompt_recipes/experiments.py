"""Parameterized recipes for improvement and experiment planning."""

from __future__ import annotations

from trainlens.llm.prompts import PromptOptions
from trainlens.prompt_recipes.base import prompt_options


def improvement_plan_prompt(
    *,
    objective: str = "Prioritize evidence-backed improvements for the next training run.",
    heading: str = "## TrainLens Improvement Plan",
    model_family: str = "machine-learning model",
    audience: str = "practitioners planning the next iteration",
    tone: str = "practical, prioritized, and evidence-first",
    rules: tuple[str, ...] = (
        "Tie every proposed change to supplied evidence.",
        "Separate low-cost checks from expensive retraining changes.",
        "Do not change multiple experimental variables without justification.",
    ),
    focus_areas: tuple[str, ...] = (
        "highest-impact changes",
        "experiment cost",
        "evidence to collect",
    ),
    return_instructions: tuple[str, ...] = (
        "Rank improvements by expected information value and cost.",
        "Include a measurable success criterion for each proposed experiment.",
    ),
) -> PromptOptions:
    """Return a fully customizable improvement-planning recipe."""

    return prompt_options(
        prompt_name="improvement_plan",
        objective=objective,
        heading=heading,
        model_family=model_family,
        audience=audience,
        tone=tone,
        rules=rules,
        focus_areas=focus_areas,
        return_instructions=return_instructions,
    )


def controlled_experiment_prompt(
    *,
    objective: str = "Design the next controlled experiment from the supplied run evidence.",
    heading: str = "## TrainLens Controlled Experiment",
    model_family: str = "machine-learning model",
    audience: str = "researchers running reproducible experiments",
    tone: str = "specific, testable, and concise",
    rules: tuple[str, ...] = (
        "Change one independent variable unless a dependency requires otherwise.",
        "Keep the control configuration explicit.",
        "Use only metrics and parameters present in the notebook context.",
    ),
    focus_areas: tuple[str, ...] = (
        "testable hypothesis",
        "controlled variable",
        "stopping and acceptance criteria",
    ),
    return_instructions: tuple[str, ...] = (
        "Return hypothesis, control, change, metrics, stopping rule, and success criterion.",
        "State the expected evidence for accepting or rejecting the hypothesis.",
    ),
) -> PromptOptions:
    """Return a fully customizable controlled-experiment recipe."""

    return prompt_options(
        prompt_name="experiment_design",
        objective=objective,
        heading=heading,
        model_family=model_family,
        audience=audience,
        tone=tone,
        rules=rules,
        focus_areas=focus_areas,
        return_instructions=return_instructions,
    )
