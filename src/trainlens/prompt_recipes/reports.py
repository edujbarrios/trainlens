"""Parameterized recipes for evidence-first training reports."""

from __future__ import annotations

from trainlens.llm.prompts import PromptOptions
from trainlens.prompt_recipes.base import prompt_options


def scientific_report_prompt(
    *,
    objective: str = "Explain the training outcome using only the supplied evidence.",
    heading: str = "## TrainLens Scientific Report",
    model_family: str = "foundation model fine-tuning",
    audience: str = "researchers and ML engineers",
    tone: str = "scientific, precise, and cautious",
    rules: tuple[str, ...] = (
        "Do not invent metrics, datasets, hyperparameters, or model names.",
        "Separate observations from interpretations and recommendations.",
        "State uncertainty whenever the supplied evidence is incomplete.",
    ),
    focus_areas: tuple[str, ...] = (
        "metric trends",
        "generalization behavior",
        "supported conclusions",
    ),
    return_instructions: tuple[str, ...] = (
        "Include Abstract, Results, Discussion, Limitations, and Next Steps sections.",
        "Preserve exact metric values from the notebook context.",
    ),
) -> PromptOptions:
    """Return a fully customizable scientific-report prompt recipe."""

    return prompt_options(
        prompt_name="scientific_report",
        objective=objective,
        heading=heading,
        model_family=model_family,
        audience=audience,
        tone=tone,
        rules=rules,
        focus_areas=focus_areas,
        return_instructions=return_instructions,
    )
