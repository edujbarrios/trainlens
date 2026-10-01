"""LLM explanation orchestration."""

from __future__ import annotations

from trainlens.llm.config import LLMConfig
from trainlens.llm.openai_compatible import OpenAICompatibleProvider
from trainlens.llm.prompts import PromptOptions, ReportMode
from trainlens.llm.provider import LLMProvider


def explain_with_llm(
    markdown_report: str,
    *,
    mode: ReportMode = "paper_report",
    require_provider: bool = False,
    prompt_options: PromptOptions | None = None,
    provider: LLMProvider | None = None,
) -> str:
    """Explain a local TrainLens report with an injected or configured provider."""

    active_provider = provider
    if active_provider is None:
        config = LLMConfig.from_env()
        if config is None:
            if require_provider:
                msg = (
                    "LLM provider configuration is missing. Set TRAINLENS_LLM_BASE_URL "
                    "and TRAINLENS_LLM_MODEL. TRAINLENS_LLM_API_KEY is optional for "
                    "local or unauthenticated endpoints."
                )
                raise RuntimeError(msg)
            return (
                markdown_report
                + "\n> LLM explanation skipped because provider configuration is missing.\n"
            )
        active_provider = OpenAICompatibleProvider(config)

    try:
        if prompt_options is None:
            return active_provider.explain(markdown_report, mode=mode)
        return active_provider.explain(
            markdown_report,
            mode=mode,
            prompt_options=prompt_options,
        )
    except Exception as exc:  # pragma: no cover - defensive notebook UX path
        if require_provider:
            msg = f"LLM explanation failed: {exc}"
            raise RuntimeError(msg) from exc
        return markdown_report + f"\n> LLM explanation failed: {exc}\n"
