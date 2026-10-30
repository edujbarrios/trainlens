"""LLM provider configuration."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from os import getenv
from pathlib import Path


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str | None = None
    model: str = "auto"
    timeout_seconds: float = 120.0

    @classmethod
    def from_env(cls) -> LLMConfig | None:
        base_url = getenv("TRAINLENS_LLM_BASE_URL", "").strip()
        api_key = getenv("TRAINLENS_LLM_API_KEY", "").strip()
        api_key_file = getenv("TRAINLENS_LLM_API_KEY_FILE", "").strip()
        model = getenv("TRAINLENS_LLM_MODEL", "").strip()
        timeout_seconds = _timeout_from_env(getenv("TRAINLENS_LLM_TIMEOUT_SECONDS", "120"))
        if not base_url or not model:
            return None
        if not api_key and api_key_file:
            api_key = _read_api_key_file(api_key_file)
        return cls(
            base_url=base_url.rstrip("/"),
            api_key=api_key or None,
            model=model,
            timeout_seconds=timeout_seconds,
        )


def _timeout_from_env(raw_value: str) -> float:
    try:
        timeout = float(raw_value.strip())
    except ValueError:
        return 120.0
    if not isfinite(timeout) or timeout <= 0:
        return 120.0
    return timeout


def _read_api_key_file(path: str) -> str:
    """Read a small mounted secret, without including its contents in errors."""

    try:
        with Path(path).open(encoding="utf-8") as file:
            key = file.read(4097)
    except (OSError, UnicodeError) as exc:
        raise ValueError("Could not read TRAINLENS_LLM_API_KEY_FILE") from exc
    if len(key) > 4096:
        raise ValueError("TRAINLENS_LLM_API_KEY_FILE exceeds 4096 characters")
    key = key.strip()
    if not key:
        raise ValueError("TRAINLENS_LLM_API_KEY_FILE is empty")
    if "\n" in key or "\r" in key:
        raise ValueError("TRAINLENS_LLM_API_KEY_FILE must contain a single key")
    return key
