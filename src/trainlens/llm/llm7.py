"""Backward-compatible import path for older TrainLens code."""

from trainlens.llm.openai_compatible import OpenAICompatibleProvider

LLM7Provider = OpenAICompatibleProvider

__all__ = ["LLM7Provider"]
