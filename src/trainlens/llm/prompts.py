"""Prompt templates for LLM explanations of ML/DL results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from jinja2 import Environment, StrictUndefined

from trainlens.security import redact_text

ReportMode = Literal["paper_report", "improvement_ideas"]


@dataclass(frozen=True)
class TrainLensPrompt:
    """A discoverable built-in prompt definition."""

    name: str
    description: str
    task: str
    heading: str
    return_instructions: tuple[str, ...]
    focus_areas: tuple[str, ...]


BUILTIN_PROMPTS: Mapping[str, TrainLensPrompt] = {
    "scientific_report": TrainLensPrompt(
        name="scientific_report",
        description="Explain a training run as an evidence-first scientific report.",
        task=(
            "generate a TrainLens paper-style Markdown report for the following DL/ML "
            "training run, with results, interpretation, and possible conclusions."
        ),
        heading="## TrainLens Scientific Report",
        return_instructions=(
            "Write in a scientific-paper style with complete paragraphs and clear sections.",
            "Use `### Abstract`, `### Methods Context`, `### Results`, `### Discussion`, "
            "`### Possible Conclusions`, and `### Limitations`.",
            "Separate observed results from possible conclusions.",
            "Explain why the observed result likely happened before suggesting changes.",
            "Include uncertainty when the notebook evidence is incomplete.",
        ),
        focus_areas=("metric trends", "validation drift", "supported conclusions"),
    ),
    "improvement_plan": TrainLensPrompt(
        name="improvement_plan",
        description="Suggest prioritized, evidence-backed improvements and experiments.",
        task=(
            "generate a TrainLens improvement plan for the following DL/ML training "
            "run, focused on follow-up experiments and evidence-backed changes."
        ),
        heading="## TrainLens Improvement Ideas",
        return_instructions=(
            "Organize the report with `### Evidence Snapshot`, `### Improvement Ideas`, "
            "`### Prioritized Experiments`, and `### Expected Evidence To Collect`.",
            "Separate low-risk notebook changes from higher-cost training changes.",
            "Explain why each idea is supported by the supplied evidence.",
            "Include uncertainty when the notebook evidence is incomplete.",
        ),
        focus_areas=("highest-impact changes", "experiment cost", "evidence to collect"),
    ),
    "training_diagnosis": TrainLensPrompt(
        name="training_diagnosis",
        description="Diagnose likely training failure modes without inventing causes.",
        task="diagnose the supplied training run and rank evidence-supported failure modes.",
        heading="## TrainLens Training Diagnosis",
        return_instructions=(
            "Separate observations, likely causes, checks, and corrective actions.",
            "Rank hypotheses by confidence and state what evidence would falsify each one.",
        ),
        focus_areas=("optimization stability", "overfitting", "data and metric anomalies"),
    ),
    "experiment_design": TrainLensPrompt(
        name="experiment_design",
        description="Turn training evidence into a controlled experiment plan.",
        task="design controlled follow-up experiments for the supplied training run.",
        heading="## TrainLens Experiment Design",
        return_instructions=(
            "Define a hypothesis, control, changed variable, metrics, and stopping rule.",
            "Prioritize experiments by information value and execution cost.",
        ),
        focus_areas=("isolated variables", "evaluation criteria", "reproducibility"),
    ),
}


def show_trainlens_prompts() -> tuple[TrainLensPrompt, ...]:
    """Return all built-in prompts in stable discovery order."""

    return tuple(BUILTIN_PROMPTS.values())


def get_trainlens_prompt(name: str) -> TrainLensPrompt:
    """Return a built-in prompt by name with a helpful error for unknown names."""

    try:
        return BUILTIN_PROMPTS[name]
    except KeyError as exc:
        available = ", ".join(BUILTIN_PROMPTS)
        raise ValueError(f"Unknown TrainLens prompt {name!r}. Available: {available}.") from exc

ML_RESULTS_EXPLANATION_TEMPLATE = """\
You are TrainLens, an evidence-first assistant for explaining ML and DL training results.

Task: {{ task }}

Audience: {{ audience }}
Tone: {{ tone }}
Target model family: {{ model_family }}
Provider mode: OpenAI-compatible chat completions
LLM model used for this report: {{ llm_model }}

Rules:
{% for rule in rules -%}
- {{ rule }}
{% endfor %}

Focus areas:
{% for focus_area in focus_areas -%}
- {{ focus_area }}
{% endfor %}

Return format:
- Start with `{{ heading }}`.
- Include a short `### LLM provenance` section naming `{{ llm_model }}` as the
  model used to draft the report.
{% for instruction in return_instructions -%}
- {{ instruction }}
{% endfor %}
- Do not mention internal heuristics or hidden analysis steps.

Notebook context:
{{ markdown_report }}
"""


@dataclass(frozen=True)
class ReportPromptContext:
    """Inputs used to render an LLM explanation prompt."""

    markdown_report: str
    mode: ReportMode = "paper_report"
    prompt_name: str | None = None
    llm_model: str = "unknown"
    model_family: str = "foundation model fine-tuning"
    audience: str = "researchers and ML engineers reviewing training results"
    tone: str = "scientific, precise, well structured, and careful"
    objective: str | None = None
    custom_heading: str | None = None
    custom_return_instructions: tuple[str, ...] | None = None
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

    @property
    def task(self) -> str:
        if self.objective is not None:
            return self.objective
        if self.prompt_name is not None:
            return get_trainlens_prompt(self.prompt_name).task
        if self.mode == "improvement_ideas":
            return (
                "generate a TrainLens improvement plan for the following DL/ML training "
                "run, focused on follow-up experiments and evidence-backed changes."
            )
        return (
            "generate a TrainLens paper-style Markdown report for the following DL/ML "
            "training run, with results, interpretation, and possible conclusions."
        )

    @property
    def heading(self) -> str:
        if self.custom_heading is not None:
            return self.custom_heading
        if self.prompt_name is not None:
            return get_trainlens_prompt(self.prompt_name).heading
        if self.mode == "improvement_ideas":
            return "## TrainLens Improvement Ideas"
        return "## TrainLens Scientific Report"

    @property
    def return_instructions(self) -> tuple[str, ...]:
        if self.custom_return_instructions is not None:
            return self.custom_return_instructions
        if self.prompt_name is not None:
            return get_trainlens_prompt(self.prompt_name).return_instructions
        if self.mode == "improvement_ideas":
            return (
                "Organize the report with `### Evidence Snapshot`, `### Improvement Ideas`, "
                "`### Prioritized Experiments`, and `### Expected Evidence To Collect`.",
                "Separate low-risk notebook changes from higher-cost training changes.",
                "Explain why each idea is supported by the supplied evidence.",
                "Include uncertainty when the notebook evidence is incomplete.",
            )
        return (
            "Write in a scientific-paper style with complete paragraphs and clear sections.",
            "Use `### Abstract`, `### Methods Context`, `### Results`, "
            "`### Discussion`, `### Possible Conclusions`, and `### Limitations`.",
            "Separate observed results from possible conclusions.",
            "Explain why the observed result likely happened before suggesting changes.",
            "Include uncertainty when the notebook evidence is incomplete.",
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
            task=context.task,
            heading=context.heading,
            return_instructions=context.return_instructions,
            llm_model=context.llm_model,
            model_family=context.model_family,
            audience=context.audience,
            tone=context.tone,
            rules=context.rules,
            focus_areas=context.focus_areas,
        )


def render_ml_results_explanation_prompt(
    markdown_report: str,
    *,
    mode: ReportMode = "paper_report",
    prompt_name: str | None = None,
    llm_model: str = "unknown",
    model_family: str = "foundation model fine-tuning",
    audience: str = "researchers and ML engineers reviewing training results",
    tone: str = "scientific, precise, well structured, and careful",
    objective: str | None = None,
    heading: str | None = None,
    return_instructions: tuple[str, ...] | None = None,
    rules: tuple[str, ...] | None = None,
    focus_areas: tuple[str, ...] | None = None,
) -> str:
    """Render the default prompt for generating an LLM-only ML/DL report."""

    base = ReportPromptContext(
        markdown_report=markdown_report,
        mode=mode,
        prompt_name=prompt_name,
        llm_model=llm_model,
        model_family=model_family,
        audience=audience,
        tone=tone,
        objective=objective,
        custom_heading=heading,
        custom_return_instructions=return_instructions,
    )
    context = ReportPromptContext(
        markdown_report=base.markdown_report,
        mode=base.mode,
        prompt_name=base.prompt_name,
        llm_model=base.llm_model,
        model_family=base.model_family,
        audience=base.audience,
        tone=base.tone,
        objective=base.objective,
        custom_heading=base.custom_heading,
        custom_return_instructions=base.custom_return_instructions,
        rules=base.rules if rules is None else rules,
        focus_areas=(
            (get_trainlens_prompt(prompt_name).focus_areas if prompt_name else base.focus_areas)
            if focus_areas is None
            else focus_areas
        ),
    )
    return ReportPromptTemplate().render(context)
