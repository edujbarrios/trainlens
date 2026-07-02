"""Provider protocol for optional LLM explanations."""

from __future__ import annotations

from typing import Protocol


class LLMProvider(Protocol):
    def explain(self, markdown_report: str) -> str:
        """Return an LLM explanation for a local TrainLens report."""
