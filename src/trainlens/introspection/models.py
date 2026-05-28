"""Introspection-specific models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelCandidate:
    """A likely trainable model discovered in notebook state."""

    variable_name: str
    object_ref: Any
    type_name: str
    module: str | None
    framework: str | None
    confidence: float
    reasons: tuple[str, ...] = ()

    @property
    def display_name(self) -> str:
        return self.type_name or self.variable_name
