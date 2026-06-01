"""Notebook convenience helpers for live TrainLens reports."""

# mypy: disable-error-code="attr-defined,no-untyped-call,no-any-return"

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from IPython import get_ipython
from IPython.display import HTML, Markdown, display

from trainlens.models.analysis import AnalysisResult
from trainlens.pipeline import explain_namespace
from trainlens.renderers.markdown import MarkdownRenderer
from trainlens.renderers.visual import DarkVisualRenderer


@dataclass(frozen=True)
class LiveReport:
    """Rendered notebook report artifacts."""

    result: AnalysisResult
    markdown: str
    dashboard_html: str


def build_live_report(namespace: Mapping[str, Any] | None = None) -> LiveReport:
    """Build Markdown and visual dashboard output from a notebook namespace."""

    report_namespace = namespace or _current_user_namespace()
    result = explain_namespace(report_namespace)
    return LiveReport(
        result=result,
        markdown=MarkdownRenderer().render(result),
        dashboard_html=DarkVisualRenderer().render_dashboard_html(result),
    )


def display_live_report(namespace: Mapping[str, Any] | None = None) -> LiveReport:
    """Display a Markdown report and dark visual dashboard in a notebook."""

    report = build_live_report(namespace)
    display(Markdown(report.markdown))
    display(HTML(report.dashboard_html))
    return report


def _current_user_namespace() -> Mapping[str, Any]:
    shell = get_ipython()
    if shell is None:
        msg = "No active IPython shell found; pass a namespace explicitly."
        raise RuntimeError(msg)
    return shell.user_ns
