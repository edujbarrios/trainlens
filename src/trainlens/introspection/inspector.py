"""Inspect active notebook state."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trainlens.introspection.frameworks import detect_framework, looks_like_model
from trainlens.introspection.models import ModelCandidate
from trainlens.models.snapshot import NotebookSnapshot, VariableInfo

_IGNORED_NAMES = {"In", "Out", "get_ipython", "exit", "quit"}


class NotebookInspector:
    """Builds a compact snapshot from an IPython user namespace."""

    def snapshot(self, namespace: Mapping[str, Any]) -> NotebookSnapshot:
        variables: list[VariableInfo] = []
        raw: dict[str, Any] = {}
        for name, value in namespace.items():
            if self._ignore(name, value):
                continue
            variables.append(self._describe(name, value))
            raw[name] = value
        return NotebookSnapshot(variables=tuple(variables), raw_namespace=raw)

    def find_models(self, snapshot: NotebookSnapshot) -> list[ModelCandidate]:
        candidates: list[ModelCandidate] = []
        for variable in snapshot.variables:
            value = snapshot.raw_namespace.get(variable.name)
            if value is None:
                continue
            looks_like, reasons = looks_like_model(value)
            framework = detect_framework(value)
            if not looks_like and framework is None:
                continue
            confidence = 0.45 + (0.2 if framework else 0) + min(len(reasons) * 0.1, 0.3)
            candidates.append(
                ModelCandidate(
                    variable_name=variable.name,
                    object_ref=value,
                    type_name=variable.type_name,
                    module=variable.module,
                    framework=framework,
                    confidence=min(confidence, 0.95),
                    reasons=reasons,
                )
            )
        return sorted(candidates, key=lambda item: item.confidence, reverse=True)

    def _ignore(self, name: str, value: object) -> bool:
        return name.startswith("_") or name in _IGNORED_NAMES or callable(value) and name.startswith("%%")

    def _describe(self, name: str, value: Any) -> VariableInfo:
        shape = getattr(value, "shape", None)
        normalized_shape = tuple(int(part) for part in shape) if shape is not None else None
        length = self._safe_len(value)
        module = getattr(value.__class__, "__module__", None)
        return VariableInfo(
            name=name,
            type_name=value.__class__.__name__,
            module=module,
            shape=normalized_shape,
            length=length,
            value=value if self._is_small_literal(value) else None,
        )

    def _safe_len(self, value: object) -> int | None:
        try:
            return len(value)  # type: ignore[arg-type]
        except TypeError:
            return None

    def _is_small_literal(self, value: object) -> bool:
        return isinstance(value, (str, int, float, bool, type(None))) or (
            isinstance(value, (list, tuple, dict, set)) and len(value) <= 20
        )
