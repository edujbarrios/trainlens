"""LLM enhancement orchestration."""

from __future__ import annotations

from trainlens.llm.config import LLMConfig
from trainlens.llm.openai_compatible import OpenAICompatibleProvider


def maybe_enhance(markdown_report: str) -> str:
    config = LLMConfig.from_env()
    if config is None:
        return (
            markdown_report
            + "\n> LLM enhancement skipped because provider configuration is missing.\n"
    )
    try:
        return OpenAICompatibleProvider(config).enhance(markdown_report)
    except Exception as exc:  # pragma: no cover - defensive notebook UX path
        return markdown_report + f"\n> LLM enhancement failed: {exc}\n"
