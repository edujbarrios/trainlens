"""Small helpers for writing readable prompt text in Python notebooks."""

from __future__ import annotations

from textwrap import dedent


def prompt_text(value: str) -> str:
    """Normalize an indented multiline prompt block.

    This is intended for triple-quoted prompt text written inside indented Python
    code. Common leading indentation and surrounding blank space are removed,
    while relative indentation inside the text is preserved.
    """

    return dedent(value).strip()
