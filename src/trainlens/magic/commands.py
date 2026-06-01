"""IPython magic commands."""

# mypy: disable-error-code="misc,no-untyped-call"

from __future__ import annotations

from typing import Any, cast

from IPython.core.magic import Magics, line_magic, magics_class
from IPython.display import HTML, Markdown, display

from trainlens.llm.enhancer import maybe_enhance
from trainlens.pipeline import explain_namespace
from trainlens.renderers.markdown import MarkdownRenderer
from trainlens.renderers.visual import DarkVisualRenderer
from trainlens.storage.memory import InMemoryRunStore


@magics_class
class TrainLensMagics(Magics):
    """Notebook commands for training explanations."""

    def __init__(self, shell: Any = None) -> None:
        super().__init__(shell)
        self.renderer = MarkdownRenderer()
        self.visual_renderer = DarkVisualRenderer()
        self.store = InMemoryRunStore()

    @line_magic
    def explain_training(self, line: str = "") -> None:
        shell = cast(Any, self.shell)
        result = explain_namespace(shell.user_ns)
        markdown = self.renderer.render(result)
        if "--llm" in line.split():
            markdown = maybe_enhance(markdown)
        self.store.capture(result)
        display(Markdown(markdown))

    @line_magic
    def training_summary(self, line: str = "") -> None:
        shell = cast(Any, self.shell)
        result = explain_namespace(shell.user_ns)
        display(Markdown(self.renderer.render(result)))

    @line_magic
    def why_bad_model(self, line: str = "") -> None:
        shell = cast(Any, self.shell)
        result = explain_namespace(shell.user_ns)
        result.summary.insert(
            0,
            "Focused diagnosis mode: prioritizing risks and next debugging steps.",
        )
        display(Markdown(self.renderer.render(result)))

    @line_magic
    def compare_runs(self, line: str = "") -> None:
        display(Markdown(self.store.render_comparison()))

    @line_magic
    def training_dashboard(self, line: str = "") -> None:
        shell = cast(Any, self.shell)
        result = explain_namespace(shell.user_ns)
        self.store.capture(result)
        display(HTML(self.visual_renderer.render_dashboard_html(result)))
