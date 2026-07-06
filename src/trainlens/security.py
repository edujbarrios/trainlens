"""Redaction helpers for notebook data that may leave the local process."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

REDACTED_VALUE = "[REDACTED]"

_SENSITIVE_NAME_PARTS = (
    "api_key",
    "apikey",
    "auth",
    "bearer",
    "client_secret",
    "connection_string",
    "credential",
    "jwt",
    "password",
    "private_key",
    "secret",
    "token",
)

_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\bgh[opsu]_[A-Za-z0-9_]{20,}\b"),
    re.compile(
        r"([?&](?:api[_-]?key|access[_-]?token|token|secret|password)=)[^&#\s]+",
        re.I,
    ),
    re.compile(r"([a-z][a-z0-9+.-]*://)[^/\s:@]+:[^/\s@]+@", re.I),
    re.compile(
        r"(?<![?&])\b(?:api[_-]?key|token|secret|password)\s*[:=]\s*['\"]?[^'\"\s,;]+",
        re.I,
    ),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{12,}\b", re.I),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
)


def is_sensitive_name(name: str) -> bool:
    """Return true when a variable or field name is likely to contain credentials."""

    normalized = name.lower().replace("-", "_")
    return any(part in normalized for part in _SENSITIVE_NAME_PARTS)


def redact_text(text: str) -> str:
    """Replace common secret shapes in free-form text."""

    redacted = text
    for pattern in _SECRET_PATTERNS:
        if pattern.groups:
            redacted = pattern.sub(
                lambda match: _redact_grouped_secret(match.group(0), match.group(1)),
                redacted,
            )
        else:
            redacted = pattern.sub(REDACTED_VALUE, redacted)
    return redacted


def _redact_grouped_secret(match_text: str, prefix: str) -> str:
    suffix = "@" if match_text.endswith("@") else ""
    return f"{prefix}{REDACTED_VALUE}{suffix}"


def sanitize_value(name: str, value: Any) -> Any:
    """Return a copy of small notebook values with sensitive content redacted."""

    if is_sensitive_name(name):
        return REDACTED_VALUE
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, Mapping):
        return {
            key: sanitize_value(str(key), nested_value)
            for key, nested_value in value.items()
        }
    if isinstance(value, tuple):
        return tuple(sanitize_value(name, item) for item in value)
    if isinstance(value, list):
        return [sanitize_value(name, item) for item in value]
    if isinstance(value, set):
        return {sanitize_value(name, item) for item in value}
    return value
