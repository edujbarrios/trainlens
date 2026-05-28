"""Provider protocol for optional explanation enhancement."""

from __future__ import annotations

from typing import Protocol


class LLMProvider(Protocol):
    def enhance(self, markdown_report: str) -> str:
        """Return an enhanced report."""
