"""High-level analysis pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trainlens.analysis_config import AnalysisConfig
from trainlens.analyzers.registry import default_registry
from trainlens.introspection import NotebookInspector
from trainlens.introspection.frameworks import detect_framework
from trainlens.introspection.models import ModelCandidate
from trainlens.models.analysis import AnalysisResult
from trainlens.models.snapshot import NotebookSnapshot


def snapshot_namespace(namespace: Mapping[str, Any]) -> NotebookSnapshot:
    """Capture one notebook snapshot for reuse across analysis/reporting paths."""

    return NotebookInspector().snapshot(namespace)


def analyze_snapshot(
    snapshot: NotebookSnapshot,
    *,
    config: AnalysisConfig | None = None,
) -> AnalysisResult:
    """Analyze an already captured notebook snapshot.

    Automatic selection remains the backwards-compatible default. Callers that
    need reproducibility can provide an :class:`AnalysisConfig` with explicit
    model/trainer/metric evidence, a registered analyzer, or strict ambiguity
    checking.
    """

    inspector = NotebookInspector()
    candidates = inspector.find_models(snapshot)
    model = _select_model(snapshot, candidates, config)
    analyzer_name = "training_session" if config is None else config.analyzer
    try:
        analyzer = default_registry().get(analyzer_name)
    except KeyError as exc:
        available = ", ".join(item.name for item in default_registry().all())
        raise ValueError(
            f"analyzer {analyzer_name!r} is not registered; available analyzers: {available}"
        ) from exc
    return analyzer.analyze(snapshot, model, config=config)


def analyze(
    namespace: Mapping[str, Any],
    *,
    config: AnalysisConfig | None = None,
) -> AnalysisResult:
    """Analyze a notebook namespace with optional explicit selection controls."""

    return analyze_snapshot(snapshot_namespace(namespace), config=config)


def explain_namespace(
    namespace: Mapping[str, Any],
    *,
    config: AnalysisConfig | None = None,
) -> AnalysisResult:
    """Backward-compatible alias for :func:`analyze`."""

    return analyze(namespace, config=config)


def _select_model(
    snapshot: NotebookSnapshot,
    candidates: list[ModelCandidate],
    config: AnalysisConfig | None,
) -> ModelCandidate | None:
    selector = config.model if config is not None else None
    if selector is not None:
        return _model_from_selector(snapshot, candidates, selector)
    if config is not None and config.strict and len(candidates) > 1:
        names = ", ".join(candidate.variable_name for candidate in candidates)
        raise ValueError(
            "multiple model candidates detected; select one explicitly with "
            f"AnalysisConfig(model=...). Candidates: {names}"
        )
    return candidates[0] if candidates else None


def _model_from_selector(
    snapshot: NotebookSnapshot,
    candidates: list[ModelCandidate],
    selector: str | object,
) -> ModelCandidate:
    if isinstance(selector, str):
        for candidate in candidates:
            if candidate.variable_name == selector:
                return candidate
        if selector not in snapshot.raw_namespace:
            raise ValueError(f"model variable {selector!r} was not found in the notebook snapshot")
        value = snapshot.raw_namespace[selector]
        return _explicit_candidate(selector, value)

    for candidate in candidates:
        if candidate.object_ref is selector:
            return candidate
    variable_name = next(
        (name for name, value in snapshot.raw_namespace.items() if value is selector),
        "<explicit-model>",
    )
    return _explicit_candidate(variable_name, selector)


def _explicit_candidate(variable_name: str, value: object) -> ModelCandidate:
    try:
        framework = detect_framework(value)
    except Exception:
        framework = None
    try:
        type_name = value.__class__.__name__
    except Exception:
        type_name = "unknown"
    try:
        module = getattr(value.__class__, "__module__", None)
    except Exception:
        module = None
    return ModelCandidate(
        variable_name=variable_name,
        object_ref=value,
        type_name=type_name,
        module=module,
        framework=framework,
        confidence=1.0,
        reasons=("explicit selection",),
    )
