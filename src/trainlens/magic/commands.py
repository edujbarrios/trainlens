"""IPython magic commands."""

# mypy: disable-error-code="misc,no-untyped-call"

from __future__ import annotations

from typing import Any, cast

from IPython.core.magic import Magics, line_magic, magics_class
from IPython.display import Markdown, display

from trainlens.llm.context import build_llm_notebook_context
from trainlens.llm.enhancer import explain_with_llm
from trainlens.models.analysis import AnalysisResult
from trainlens.storage.memory import InMemoryRunStore


@magics_class
class TrainLensMagics(Magics):
    """Notebook commands for training explanations."""

    def __init__(self, shell: Any = None) -> None:
        super().__init__(shell)
        self.store = InMemoryRunStore()

    @line_magic
    def explain_training(self, line: str = "") -> None:
        shell = cast(Any, self.shell)
        context = build_llm_notebook_context(shell.user_ns)
        result = AnalysisResult(metrics=context.metrics)
        markdown = explain_with_llm(
            context.markdown,
            mode="paper_report",
            require_provider=True,
        )
        self.store.capture(result)
        display(Markdown(markdown))

    @line_magic
    def suggest_improvements(self, line: str = "") -> None:
        shell = cast(Any, self.shell)
        context = build_llm_notebook_context(shell.user_ns)
        markdown = explain_with_llm(
            context.markdown,
            mode="improvement_ideas",
            require_provider=True,
        )
        display(Markdown(markdown))

    @line_magic
    def compare_runs(self, line: str = "") -> None:
        display(Markdown(self.store.render_comparison()))
