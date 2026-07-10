"""Inspect active notebook state."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

from trainlens.introspection.adapters import extract_framework_artifact
from trainlens.introspection.frameworks import detect_framework, looks_like_model
from trainlens.introspection.models import ModelCandidate
from trainlens.models.snapshot import FrameworkArtifact, NotebookSnapshot, VariableInfo
from trainlens.security import sanitize_value

_IGNORED_NAMES = {"In", "Out", "get_ipython", "exit", "quit"}


class NotebookInspector:
    """Builds a compact snapshot from an IPython user namespace."""

    def snapshot(self, namespace: Mapping[str, Any]) -> NotebookSnapshot:
        variables: list[VariableInfo] = []
        framework_artifacts: list[FrameworkArtifact] = []
        raw: dict[str, Any] = {}
        for name, value in namespace.items():
            if self._ignore(name, value):
                continue
            variables.append(self._describe(name, value))
            artifact = extract_framework_artifact(name, value)
            if artifact:
                framework_artifacts.append(artifact)
            raw[name] = value
        return NotebookSnapshot(
            variables=tuple(variables),
            framework_artifacts=tuple(framework_artifacts),
            raw_namespace=raw,
        )

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
        candidates.extend(self._framework_model_candidates(snapshot))
        return sorted(candidates, key=lambda item: item.confidence, reverse=True)

    def _framework_model_candidates(self, snapshot: NotebookSnapshot) -> list[ModelCandidate]:
        candidates: list[ModelCandidate] = []
        for artifact in snapshot.framework_artifacts:
            if artifact.model_ref is None and artifact.model_name is None:
                continue
            model_ref = artifact.model_ref or snapshot.raw_namespace.get(artifact.variable_name)
            type_name = artifact.model_name or artifact.type_name
            candidates.append(
                ModelCandidate(
                    variable_name=artifact.variable_name,
                    object_ref=model_ref,
                    type_name=type_name,
                    module=getattr(model_ref.__class__, "__module__", None),
                    framework=artifact.framework,
                    confidence=min(artifact.confidence + 0.05, 0.95),
                    reasons=artifact.reasons,
                )
            )
        return candidates

    def _ignore(self, name: str, value: object) -> bool:
        return (
            name.startswith("_")
            or name in _IGNORED_NAMES
            or inspect.isclass(value)
            or inspect.isfunction(value)
            or inspect.ismodule(value)
            or callable(value)
            and name.startswith("%%")
        )

    def _describe(self, name: str, value: Any) -> VariableInfo:
        shape = getattr(value, "shape", None)
        normalized_shape = self._normalize_shape(shape)
        length = self._safe_len(value)
        module = getattr(value.__class__, "__module__", None)
        return VariableInfo(
            name=name,
            type_name=value.__class__.__name__,
            module=module,
            shape=normalized_shape,
            length=length,
            value=sanitize_value(name, value) if self._is_small_literal(value) else None,
        )

    def _safe_len(self, value: object) -> int | None:
        try:
            return len(value)  # type: ignore[arg-type]
        except TypeError:
            return None

    def _normalize_shape(self, shape: Any) -> tuple[int | None, ...] | None:
        if shape is None:
            return None
        try:
            return tuple(self._normalize_shape_part(part) for part in shape)
        except TypeError:
            return None

    def _normalize_shape_part(self, part: Any) -> int | None:
        if part is None:
            return None
        try:
            return int(part)
        except (TypeError, ValueError):
            return None

    def _is_small_literal(self, value: object) -> bool:
        return isinstance(value, str | int | float | bool | type(None)) or (
            isinstance(value, list | tuple | dict | set) and len(value) <= 20
        )
