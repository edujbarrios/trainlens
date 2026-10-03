"""Optional LLM explanations."""

from trainlens.llm.config import LLMConfig
from trainlens.llm.enhancer import explain_with_llm
from trainlens.llm.openai_compatible import LLMRequestPreview, OpenAICompatibleProvider
from trainlens.llm.prompts import (
    PromptOptions,
    ReportPromptContext,
    ReportPromptTemplate,
    TrainLensPrompt,
    get_trainlens_prompt,
    show_trainlens_prompts,
)
from trainlens.llm.provider import LLMProvider

__all__ = [
    "LLMConfig",
    "LLMProvider",
    "LLMRequestPreview",
    "OpenAICompatibleProvider",
    "PromptOptions",
    "ReportPromptContext",
    "ReportPromptTemplate",
    "TrainLensPrompt",
    "explain_with_llm",
    "get_trainlens_prompt",
    "show_trainlens_prompts",
]
