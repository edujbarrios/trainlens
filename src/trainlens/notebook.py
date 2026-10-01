"""Notebook convenience helpers for TrainLens LLM reports."""

# mypy: disable-error-code="attr-defined,no-untyped-call,no-any-return"

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from IPython import get_ipython

from trainlens.analysis_config import AnalysisConfig
from trainlens.llm.context import (
    ContextPolicy,
    LLMNotebookContext,
    build_llm_notebook_context_from_snapshot,
)
from trainlens.llm.enhancer import explain_with_llm
from trainlens.llm.prompts import PromptOptions, ReportMode
from trainlens.llm.provider import LLMProvider
from trainlens.models.analysis import AnalysisResult
from trainlens.pipeline import analyze_snapshot, snapshot_namespace


@dataclass(frozen=True)
class LiveReport:
    """Rendered notebook report artifacts."""

    result: AnalysisResult
    markdown: str

    def _repr_markdown_(self) -> str:
        return self.markdown


def preview_notebook_context(
    namespace: Mapping[str, Any] | None = None,
    *,
    max_metric_points: int = 12,
    include_values: bool = False,
    analysis_config: AnalysisConfig | None = None,
    context_policy: ContextPolicy | None = None,
) -> LLMNotebookContext:
    """Return the exact sanitized notebook context used for an LLM request."""

    report_namespace = _current_user_namespace() if namespace is None else namespace
    snapshot = snapshot_namespace(report_namespace)
    result = analyze_snapshot(snapshot, config=analysis_config)
    return build_llm_notebook_context_from_snapshot(
        snapshot,
        max_metric_points=max_metric_points,
        include_values=include_values,
        analysis_config=analysis_config,
        deterministic_result=result,
        context_policy=context_policy,
    )


def build_llm_report(
    namespace: Mapping[str, Any] | None = None,
    *,
    max_metric_points: int = 12,
    prompt_options: PromptOptions | None = None,
    include_values: bool = False,
    analysis_config: AnalysisConfig | None = None,
    provider: LLMProvider | None = None,
    context_policy: ContextPolicy | None = None,
) -> LiveReport:
    """Build an LLM-generated training report from notebook context."""

    return build_paper_report(
        namespace,
        max_metric_points=max_metric_points,
        prompt_options=prompt_options,
        include_values=include_values,
        analysis_config=analysis_config,
        provider=provider,
        context_policy=context_policy,
    )


def build_paper_report(
    namespace: Mapping[str, Any] | None = None,
    *,
    max_metric_points: int = 12,
    prompt_options: PromptOptions | None = None,
    include_values: bool = False,
    analysis_config: AnalysisConfig | None = None,
    provider: LLMProvider | None = None,
    context_policy: ContextPolicy | None = None,
) -> LiveReport:
    """Build a scientific paper-style training report from notebook context."""

    return _build_report(
        namespace,
        mode="paper_report",
        max_metric_points=max_metric_points,
        prompt_options=prompt_options,
        include_values=include_values,
        analysis_config=analysis_config,
        provider=provider,
        context_policy=context_policy,
    )


def build_improvement_ideas(
    namespace: Mapping[str, Any] | None = None,
    *,
    max_metric_points: int = 12,
    prompt_options: PromptOptions | None = None,
    include_values: bool = False,
    analysis_config: AnalysisConfig | None = None,
    provider: LLMProvider | None = None,
    context_policy: ContextPolicy | None = None,
) -> LiveReport:
    """Build an evidence-backed improvement plan from notebook context."""

    return _build_report(
        namespace,
        mode="improvement_ideas",
        max_metric_points=max_metric_points,
        prompt_options=prompt_options,
        include_values=include_values,
        analysis_config=analysis_config,
        provider=provider,
        context_policy=context_policy,
    )


def _build_report(
    namespace: Mapping[str, Any] | None,
    *,
    mode: ReportMode,
    max_metric_points: int,
    prompt_options: PromptOptions | None,
    include_values: bool,
    analysis_config: AnalysisConfig | None,
    provider: LLMProvider | None,
    context_policy: ContextPolicy | None,
) -> LiveReport:
    report_namespace = _current_user_namespace() if namespace is None else namespace
    snapshot = snapshot_namespace(report_namespace)
    result = analyze_snapshot(snapshot, config=analysis_config)
    context = build_llm_notebook_context_from_snapshot(
        snapshot,
        max_metric_points=max_metric_points,
        include_values=include_values,
        analysis_config=analysis_config,
        deterministic_result=result,
        context_policy=context_policy,
    )
    explain_kwargs: dict[str, Any] = {"mode": mode, "require_provider": True}
    if prompt_options is not None:
        explain_kwargs["prompt_options"] = prompt_options
    if provider is not None:
        explain_kwargs["provider"] = provider
    return LiveReport(
        result=result,
        markdown=explain_with_llm(context.markdown, **explain_kwargs),
    )


def _current_user_namespace() -> Mapping[str, Any]:
    shell = get_ipython()
    if shell is None:
        raise RuntimeError("No active IPython shell found; pass a namespace explicitly.")
    return shell.user_ns
