"""Shared helpers for reusable, fully parameterized prompt recipes."""

from __future__ import annotations

from trainlens.llm.prompts import PromptOptions, get_trainlens_prompt


def prompt_options(
    *,
    prompt_name: str,
    objective: str,
    heading: str,
    model_family: str,
    audience: str,
    tone: str,
    rules: tuple[str, ...],
    focus_areas: tuple[str, ...],
    return_instructions: tuple[str, ...],
) -> PromptOptions:
    """Build prompt options while requiring every configurable field."""

    get_trainlens_prompt(prompt_name)
    return PromptOptions(
        prompt_name=prompt_name,
        objective=objective,
        heading=heading,
        model_family=model_family,
        audience=audience,
        tone=tone,
        rules=rules,
        focus_areas=focus_areas,
        return_instructions=return_instructions,
    )
