"""LLM explanation orchestration."""

from __future__ import annotations

from trainlens.llm.config import LLMConfig
from trainlens.llm.openai_compatible import OpenAICompatibleProvider


def explain_with_llm(markdown_report: str) -> str:
    """Explain a local TrainLens report with the configured LLM provider."""

    config = LLMConfig.from_env()
    if config is None:
        return (
            markdown_report
            + "\n> LLM explanation skipped because provider configuration is missing.\n"
        )
    try:
        return OpenAICompatibleProvider(config).explain(markdown_report)
    except Exception as exc:  # pragma: no cover - defensive notebook UX path
        return markdown_report + f"\n> LLM explanation failed: {exc}\n"


def maybe_enhance(markdown_report: str) -> str:
    """Backward-compatible alias for older notebooks."""

    return explain_with_llm(markdown_report)
