"""Rich console rendering fallback."""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown


class RichRenderer:
    """Display Markdown through Rich when IPython display is unavailable."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def display(self, markdown: str) -> None:
        self.console.print(Markdown(markdown))
