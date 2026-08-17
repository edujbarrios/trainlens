"""Parameterized recipes for diagnosing training behavior."""

from __future__ import annotations

from trainlens.llm.prompts import PromptOptions
from trainlens.prompt_recipes.base import prompt_options


def training_diagnosis_prompt(
    *,
    objective: str = "Rank the most likely causes of the observed training behavior.",
    heading: str = "## TrainLens Training Diagnosis",
    model_family: str = "deep-learning model",
    audience: str = "ML engineers debugging a training run",
    tone: str = "technical, direct, and uncertainty-aware",
    rules: tuple[str, ...] = (
        "Use only evidence present in the notebook context.",
        "Do not present correlation as a confirmed cause.",
        "State what evidence would falsify each hypothesis.",
    ),
    focus_areas: tuple[str, ...] = (
        "optimization stability",
        "generalization gap",
        "data and metric anomalies",
    ),
    return_instructions: tuple[str, ...] = (
        "Rank hypotheses by confidence.",
        "For each hypothesis, provide evidence, a verification check, and a corrective action.",
    ),
) -> PromptOptions:
    """Return a fully customizable general training-diagnosis recipe."""

    return prompt_options(
        prompt_name="training_diagnosis",
        objective=objective,
        heading=heading,
        model_family=model_family,
        audience=audience,
        tone=tone,
        rules=rules,
        focus_areas=focus_areas,
        return_instructions=return_instructions,
    )


def overfitting_review_prompt(
    *,
    objective: str = "Determine whether the run shows evidence of overfitting.",
    heading: str = "## TrainLens Overfitting Review",
    model_family: str = "supervised learning model",
    audience: str = "researchers reviewing generalization quality",
    tone: str = "evidence-first, quantitative, and cautious",
    rules: tuple[str, ...] = (
        "Compare training and validation evidence over the same steps.",
        "Do not diagnose overfitting from a single metric value.",
        "Distinguish missing evidence from evidence against overfitting.",
    ),
    focus_areas: tuple[str, ...] = (
        "training-validation divergence",
        "best validation checkpoint",
        "regularization evidence",
    ),
    return_instructions: tuple[str, ...] = (
        "Return a verdict with confidence and supporting metric values.",
        "Propose the smallest controlled follow-up experiment.",
    ),
) -> PromptOptions:
    """Return a fully customizable overfitting-review recipe."""

    return prompt_options(
        prompt_name="training_diagnosis",
        objective=objective,
        heading=heading,
        model_family=model_family,
        audience=audience,
        tone=tone,
        rules=rules,
        focus_areas=focus_areas,
        return_instructions=return_instructions,
    )
