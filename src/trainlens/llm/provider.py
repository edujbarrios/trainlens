"""Provider protocol for optional LLM explanations."""

from __future__ import annotations

from typing import Protocol

from trainlens.llm.prompts import PromptOptions, ReportMode


class LLMProvider(Protocol):
    """Transport abstraction for TrainLens LLM report generation."""

    def explain(
        self,
        markdown_report: str,
        *,
        mode: ReportMode = "paper_report",
        prompt_options: PromptOptions | None = None,
    ) -> str:
        """Return an LLM explanation for a local TrainLens report."""
