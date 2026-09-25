"""Notebook convenience helpers for TrainLens LLM reports."""

# mypy: disable-error-code="attr-defined,no-untyped-call,no-any-return"

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from IPython import get_ipython

from trainlens.llm.context import LLMNotebookContext, build_llm_notebook_context
from trainlens.llm.enhancer import explain_with_llm
from trainlens.llm.prompts import PromptOptions, ReportMode
from trainlens.models.analysis import AnalysisResult
from trainlens.pipeline import explain_namespace


@dataclass(frozen=True)
class LiveReport:
    """Rendered notebook report artifacts."""

    result: AnalysisResult
    markdown: str

    def _repr_markdown_(self) -> str:
        """Render the report when returned as the last expression in IPython."""

        return self.markdown


def preview_notebook_context(
    namespace: Mapping[str, Any] | None = None,
    *,
    max_metric_points: int = 12,
    include_values: bool = False,
) -> LLMNotebookContext:
    """Return the exact sanitized notebook context used for an LLM request."""

    report_namespace = _current_user_namespace() if namespace is None else namespace
    return build_llm_notebook_context(
        report_namespace,
        max_metric_points=max_metric_points,
        include_values=include_values,
    )


def build_llm_report(
    namespace: Mapping[str, Any] | None = None,
    *,
    max_metric_points: int = 12,
    prompt_options: PromptOptions | None = None,
    include_values: bool = False,
) -> LiveReport:
    """Build an LLM-generated training report from notebook context."""

    return build_paper_report(
        namespace,
        max_metric_points=max_metric_points,
        prompt_options=prompt_options,
        include_values=include_values,
    )


def build_paper_report(
    namespace: Mapping[str, Any] | None = None,
    *,
    max_metric_points: int = 12,
    prompt_options: PromptOptions | None = None,
    include_values: bool = False,
) -> LiveReport:
    """Build a scientific paper-style training report from notebook context."""

    return _build_report(
        namespace,
        mode="paper_report",
        max_metric_points=max_metric_points,
        prompt_options=prompt_options,
        include_values=include_values,
    )


def build_improvement_ideas(
    namespace: Mapping[str, Any] | None = None,
    *,
    max_metric_points: int = 12,
    prompt_options: PromptOptions | None = None,
    include_values: bool = False,
) -> LiveReport:
    """Build an evidence-backed improvement plan from notebook context."""

    return _build_report(
        namespace,
        mode="improvement_ideas",
        max_metric_points=max_metric_points,
        prompt_options=prompt_options,
        include_values=include_values,
    )


def _build_report(
    namespace: Mapping[str, Any] | None,
    *,
    mode: ReportMode,
    max_metric_points: int,
    prompt_options: PromptOptions | None,
    include_values: bool,
) -> LiveReport:
    """Build one of the supported LLM-generated report modes."""

    report_namespace = _current_user_namespace() if namespace is None else namespace
    result = explain_namespace(report_namespace)
    context = build_llm_notebook_context(
        report_namespace,
        max_metric_points=max_metric_points,
        include_values=include_values,
    )
    explain_kwargs: dict[str, Any] = {"mode": mode, "require_provider": True}
    if prompt_options is not None:
        explain_kwargs["prompt_options"] = prompt_options
    return LiveReport(
        result=result,
        markdown=explain_with_llm(context.markdown, **explain_kwargs),
    )


def _current_user_namespace() -> Mapping[str, Any]:
    shell = get_ipython()
    if shell is None:
        msg = "No active IPython shell found; pass a namespace explicitly."
        raise RuntimeError(msg)
    return shell.user_ns
