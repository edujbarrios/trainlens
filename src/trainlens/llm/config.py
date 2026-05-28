"""LLM provider configuration."""

from __future__ import annotations

from dataclasses import dataclass
from os import getenv


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str = "auto"

    @classmethod
    def from_env(cls) -> LLMConfig | None:
        base_url = getenv("TRAINLENS_LLM_BASE_URL")
        api_key = getenv("TRAINLENS_LLM_API_KEY")
        model = getenv("TRAINLENS_LLM_MODEL", "auto")
        if not base_url or not api_key:
            return None
        return cls(base_url=base_url.rstrip("/"), api_key=api_key, model=model)
