"""Small helpers for writing and loading readable prompt text."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent


def prompt_text(value: str) -> str:
    """Normalize an indented multiline prompt block.

    This is intended for triple-quoted prompt text written inside indented Python
    code. Common leading indentation and surrounding blank space are removed,
    while relative indentation inside the text is preserved.
    """

    return dedent(value).strip()


def load_prompt(path: str | Path) -> str:
    """Load prompt text from a UTF-8 Markdown file.

    Markdown structure and relative indentation are preserved exactly; only outer
    whitespace is removed. The explicit ``.md`` check catches accidental loading of
    unrelated files early while keeping prompt files easy to review and version.
    """

    prompt_path = Path(path)
    if prompt_path.suffix.lower() != ".md":
        raise ValueError("load_prompt() expects a .md file")

    return prompt_path.read_text(encoding="utf-8").strip()
