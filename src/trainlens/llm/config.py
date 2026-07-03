"""LLM provider configuration."""

from __future__ import annotations

from dataclasses import dataclass
from os import getenv


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str = "auto"
    timeout_seconds: float = 120.0

    @classmethod
    def from_env(cls) -> LLMConfig | None:
        base_url = getenv("TRAINLENS_LLM_BASE_URL", "").strip()
        api_key = getenv("TRAINLENS_LLM_API_KEY", "").strip()
        model = getenv("TRAINLENS_LLM_MODEL", "auto").strip() or "auto"
        timeout_seconds = _timeout_from_env(getenv("TRAINLENS_LLM_TIMEOUT_SECONDS", "120"))
        if not base_url or not api_key:
            return None
        return cls(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            model=model,
            timeout_seconds=timeout_seconds,
        )


def _timeout_from_env(raw_value: str) -> float:
    try:
        timeout = float(raw_value.strip())
    except ValueError:
        return 120.0
    if timeout <= 0:
        return 120.0
    return timeout
