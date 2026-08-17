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


def sensitivity_analysis_prompt(
    *,
    objective: str = "Design a sensitivity analysis for the selected training parameters.",
    heading: str = "## TrainLens Sensitivity Analysis",
    model_family: str = "machine-learning model",
    audience: str = "researchers assessing parameter robustness",
    tone: str = "quantitative, structured, and uncertainty-aware",
    rules: tuple[str, ...] = (
        "Define parameter ranges and sampling scales before evaluating outcomes.",
        "Keep non-tested parameters and evaluation data fixed.",
        "Report instability, monotonicity, and interaction evidence separately.",
    ),
    focus_areas: tuple[str, ...] = (
        "parameter response curves",
        "robust operating regions",
        "interaction and threshold effects",
    ),
    return_instructions: tuple[str, ...] = (
        "Return parameters, ranges, sampling design, metrics, and robustness criteria.",
        "Specify plots or summary statistics needed to interpret sensitivity.",
    ),
) -> PromptOptions:
    """Return a fully customizable sensitivity-analysis recipe."""

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


def reproducibility_audit_prompt(
    *,
    objective: str = "Audit whether the training result can be independently reproduced.",
    heading: str = "## TrainLens Reproducibility Audit",
    model_family: str = "machine-learning model",
    audience: str = "reviewers reproducing the reported experiment",
    tone: str = "methodical, explicit, and audit-ready",
    rules: tuple[str, ...] = (
        "Mark missing configuration as unavailable instead of inferring it.",
        "Separate deterministic controls from sources of stochastic variation.",
        "Require repeated-run evidence for claims of stability.",
    ),
    focus_areas: tuple[str, ...] = (
        "data and code provenance",
        "seeds and deterministic settings",
        "environment and dependency capture",
    ),
    return_instructions: tuple[str, ...] = (
        "Return a pass, partial, or fail assessment for each reproducibility dimension.",
        "Provide a minimal replication protocol and list every missing artifact.",
    ),
) -> PromptOptions:
    """Return a fully customizable reproducibility-audit recipe."""

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
