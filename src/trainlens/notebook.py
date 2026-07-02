"""Notebook convenience helpers for live TrainLens reports."""

# mypy: disable-error-code="attr-defined,no-untyped-call,no-any-return"

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from IPython import get_ipython
from IPython.display import Markdown, display

from trainlens.llm.context import build_llm_notebook_context
from trainlens.llm.enhancer import explain_with_llm
from trainlens.models.analysis import AnalysisResult
from trainlens.pipeline import explain_namespace
from trainlens.renderers.markdown import MarkdownRenderer


@dataclass(frozen=True)
class LiveReport:
    """Rendered notebook report artifacts."""

    result: AnalysisResult
    markdown: str


def build_live_report(namespace: Mapping[str, Any] | None = None) -> LiveReport:
    """Build Markdown training-result output from a notebook namespace."""

    report_namespace = _current_user_namespace() if namespace is None else namespace
    result = explain_namespace(report_namespace)
    return LiveReport(
        result=result,
        markdown=MarkdownRenderer().render(result),
    )


def display_live_report(namespace: Mapping[str, Any] | None = None) -> LiveReport:
    """Display a Markdown training-result report in a notebook."""

    report = build_live_report(namespace)
    display(Markdown(report.markdown))
    return report


def build_llm_report(namespace: Mapping[str, Any] | None = None) -> LiveReport:
    """Build an LLM-generated training report from notebook context."""

    report_namespace = _current_user_namespace() if namespace is None else namespace
    context = build_llm_notebook_context(report_namespace)
    return LiveReport(
        result=AnalysisResult(metrics=context.metrics),
        markdown=explain_with_llm(context.markdown, require_provider=True),
    )


def display_llm_report(namespace: Mapping[str, Any] | None = None) -> LiveReport:
    """Display an LLM-explained training report in a notebook."""

    report = build_llm_report(namespace)
    display(Markdown(report.markdown))
    return report


def _current_user_namespace() -> Mapping[str, Any]:
    shell = get_ipython()
    if shell is None:
        msg = "No active IPython shell found; pass a namespace explicitly."
        raise RuntimeError(msg)
    return shell.user_ns
