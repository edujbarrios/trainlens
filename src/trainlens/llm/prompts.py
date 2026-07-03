"""Prompt templates for LLM explanations of ML/DL results."""

from __future__ import annotations

from dataclasses import dataclass, field

from jinja2 import Environment, StrictUndefined

from trainlens.security import redact_text

ML_RESULTS_EXPLANATION_TEMPLATE = """\
You are TrainLens, an evidence-first assistant for explaining ML and DL training results.

Task: generate a TrainLens Markdown report for the following DL/ML training run.

Audience: {{ audience }}
Tone: {{ tone }}
Target model family: {{ model_family }}
Provider mode: OpenAI-compatible chat completions

Rules:
{% for rule in rules -%}
- {{ rule }}
{% endfor %}

Focus areas:
{% for focus_area in focus_areas -%}
- {{ focus_area }}
{% endfor %}

Return format:
- Start with `## TrainLens Report`.
- Prefer short, actionable bullets over generic advice.
- Separate evidence from recommendations.
- Explain why the observed result likely happened before suggesting changes.
- Include uncertainty when the notebook evidence is incomplete.
- Do not mention internal heuristics or hidden analysis steps.

Notebook context:
{{ markdown_report }}
"""


@dataclass(frozen=True)
class ReportPromptContext:
    """Inputs used to render an LLM explanation prompt."""

    markdown_report: str
    model_family: str = "foundation model fine-tuning"
    audience: str = "ML engineers debugging notebook training runs"
    tone: str = "concise, technical, and careful"
    rules: tuple[str, ...] = (
        "Do not invent metrics, datasets, hyperparameters, or model names.",
        "Only infer risks that are supported by the notebook context.",
        "Preserve exact metric values from the context.",
        "Do not claim API access, training access, or hidden notebook state.",
        "Treat redacted placeholders as intentionally unavailable private data.",
    )
    focus_areas: tuple[str, ...] = field(
        default_factory=lambda: (
            "loss trends and validation drift",
            "LLM, CLIP, ViT, projector, and VLM fine-tuning risks",
            "adapter capacity, trainable parameter ratio, and frozen modules",
            "next experiments that can be run in the notebook",
        )
    )


class ReportPromptTemplate:
    """Parameterized Jinja2 template for explaining ML/DL results."""

    def __init__(self, template: str = ML_RESULTS_EXPLANATION_TEMPLATE) -> None:
        self._template = Environment(
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=StrictUndefined,
        ).from_string(template)

    def render(self, context: ReportPromptContext) -> str:
        return self._template.render(
            markdown_report=redact_text(context.markdown_report),
            model_family=context.model_family,
            audience=context.audience,
            tone=context.tone,
            rules=context.rules,
            focus_areas=context.focus_areas,
        )


def render_ml_results_explanation_prompt(
    markdown_report: str,
    *,
    model_family: str = "foundation model fine-tuning",
    audience: str = "ML engineers debugging notebook training runs",
    tone: str = "concise, technical, and careful",
    rules: tuple[str, ...] | None = None,
    focus_areas: tuple[str, ...] | None = None,
) -> str:
    """Render the default prompt for generating an LLM-only ML/DL report."""

    base = ReportPromptContext(
        markdown_report=markdown_report,
        model_family=model_family,
        audience=audience,
        tone=tone,
    )
    context = ReportPromptContext(
        markdown_report=base.markdown_report,
        model_family=base.model_family,
        audience=base.audience,
        tone=base.tone,
        rules=rules or base.rules,
        focus_areas=focus_areas or base.focus_areas,
    )
    return ReportPromptTemplate().render(context)
