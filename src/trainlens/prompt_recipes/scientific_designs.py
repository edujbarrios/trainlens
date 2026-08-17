"""Parameterized recipes for rigorous scientific experiment designs."""

from __future__ import annotations

from trainlens.llm.prompts import PromptOptions
from trainlens.prompt_recipes.base import prompt_options


def hypothesis_test_prompt(
    *,
    objective: str = "Design a falsifiable hypothesis test from the supplied training evidence.",
    heading: str = "## TrainLens Hypothesis Test",
    model_family: str = "machine-learning model",
    audience: str = "researchers evaluating a causal training hypothesis",
    tone: str = "formal, quantitative, and falsifiable",
    rules: tuple[str, ...] = (
        "Define the null and alternative hypotheses before interpreting evidence.",
        "Specify independent, dependent, and controlled variables.",
        "Do not claim statistical significance without an explicit test and sample size.",
    ),
    focus_areas: tuple[str, ...] = (
        "falsifiable hypothesis",
        "measurement validity",
        "confounders and controls",
    ),
    return_instructions: tuple[str, ...] = (
        "Return hypotheses, variables, protocol, analysis method, and decision rule.",
        "Include assumptions, threats to validity, and required replication evidence.",
    ),
) -> PromptOptions:
    """Return a fully customizable hypothesis-testing recipe."""

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


def ablation_study_prompt(
    *,
    objective: str = "Design an ablation study that isolates component contributions.",
    heading: str = "## TrainLens Ablation Study",
    model_family: str = "modular machine-learning system",
    audience: str = "researchers measuring component-level effects",
    tone: str = "systematic, controlled, and quantitative",
    rules: tuple[str, ...] = (
        "Change one component at a time against a shared baseline.",
        "Keep data splits, seeds, training budgets, and evaluation procedures fixed.",
        "Distinguish interaction effects from individual component effects.",
    ),
    focus_areas: tuple[str, ...] = (
        "component contribution",
        "interaction effects",
        "compute-normalized comparison",
    ),
    return_instructions: tuple[str, ...] = (
        "Return an ablation matrix with controls, variants, metrics, and expected evidence.",
        "Specify replication count, aggregation method, and acceptance thresholds.",
    ),
) -> PromptOptions:
    """Return a fully customizable ablation-study recipe."""

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
