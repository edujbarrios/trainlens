"""Shared semantic rules for normalized training metrics."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

MetricDirection = Literal["lower", "higher"]


@dataclass(frozen=True)
class MetricSpec:
    """Optimization semantics for one metric family."""

    name: str
    direction: MetricDirection
    lower_bound: float | None = None
    upper_bound: float | None = None
    aliases: tuple[str, ...] = ()
    unit: str | None = None
    material_relative_delta: float | None = None
    material_absolute_delta: float | None = None


@dataclass(frozen=True)
class MetricSemantics:
    """Known optimization direction and optional natural bounds for a metric."""

    direction: MetricDirection
    lower_bound: float | None = None
    upper_bound: float | None = None


class MetricRegistry:
    """Mutable registry used by comparison and experiment-planning helpers."""

    def __init__(self) -> None:
        self._specs: dict[str, MetricSpec] = {}
        self._aliases: dict[str, str] = {}

    def register(self, spec: MetricSpec, *, replace: bool = False) -> None:
        canonical = _normalize_token(spec.name)
        if not canonical:
            raise ValueError("metric name cannot be empty")
        if canonical in self._specs and not replace:
            raise ValueError(f"metric {spec.name!r} is already registered")
        if replace and canonical in self._specs:
            self.unregister(canonical)
        normalized_aliases = tuple(
            alias for alias in (_normalize_token(item) for item in spec.aliases) if alias
        )
        for alias in (canonical, *normalized_aliases):
            existing = self._aliases.get(alias)
            if existing is not None and existing != canonical and not replace:
                raise ValueError(f"metric alias {alias!r} is already registered")
        normalized = MetricSpec(
            name=canonical,
            direction=spec.direction,
            lower_bound=spec.lower_bound,
            upper_bound=spec.upper_bound,
            aliases=normalized_aliases,
            unit=spec.unit,
            material_relative_delta=spec.material_relative_delta,
            material_absolute_delta=spec.material_absolute_delta,
        )
        self._specs[canonical] = normalized
        for alias in (canonical, *normalized_aliases):
            self._aliases[alias] = canonical

    def unregister(self, name: str) -> None:
        canonical = self._aliases.get(_normalize_token(name), _normalize_token(name))
        spec = self._specs.pop(canonical, None)
        if spec is None:
            return
        for alias in (canonical, *spec.aliases):
            if self._aliases.get(alias) == canonical:
                del self._aliases[alias]

    def resolve(self, name: str) -> MetricSpec | None:
        normalized = _normalize_token(name)
        canonical = self._aliases.get(normalized)
        if canonical is not None:
            return self._specs[canonical]
        for token in _metric_tokens(name):
            canonical = self._aliases.get(token)
            if canonical is not None:
                return self._specs[canonical]
        return None

    def specs(self) -> tuple[MetricSpec, ...]:
        """Return registered specs in insertion order."""

        return tuple(self._specs.values())


_DEFAULT_REGISTRY = MetricRegistry()


def _register_defaults() -> None:
    lower = (
        "loss",
        "error",
        "perplexity",
        "wer",
        "cer",
        "latency",
        "fad",
        "frechet",
        "mae",
        "mape",
        "mse",
        "msle",
        "rmse",
    )
    bounded_higher = ("accuracy", "acc", "auc", "f1", "precision", "recall", "map", "ndcg")
    for name in lower:
        _DEFAULT_REGISTRY.register(
            MetricSpec(name=name, direction="lower", lower_bound=0.0)
        )
    for name in bounded_higher:
        _DEFAULT_REGISTRY.register(
            MetricSpec(name=name, direction="higher", lower_bound=0.0, upper_bound=1.0)
        )
    _DEFAULT_REGISTRY.register(MetricSpec(name="score", direction="higher"))


_register_defaults()


def default_metric_registry() -> MetricRegistry:
    """Return the process-wide registry used by TrainLens semantic helpers."""

    return _DEFAULT_REGISTRY


def register_metric(spec: MetricSpec, *, replace: bool = False) -> None:
    """Register custom semantics for subsequent TrainLens operations."""

    _DEFAULT_REGISTRY.register(spec, replace=replace)


def unregister_metric(name: str) -> None:
    """Remove one custom metric family from the default registry."""

    _DEFAULT_REGISTRY.unregister(name)


def metric_semantics(name: str) -> MetricSemantics | None:
    """Return shared semantics inferred from the registered metric families."""

    spec = _DEFAULT_REGISTRY.resolve(name)
    if spec is None:
        return None
    return MetricSemantics(
        direction=spec.direction,
        lower_bound=spec.lower_bound,
        upper_bound=spec.upper_bound,
    )


def metric_direction(name: str) -> MetricDirection | None:
    """Return whether a known metric should move lower or higher."""

    semantics = metric_semantics(name)
    return semantics.direction if semantics is not None else None


def metric_bounds(name: str) -> tuple[float | None, float | None]:
    """Return known natural bounds, or ``(None, None)`` for unknown metrics."""

    semantics = metric_semantics(name)
    if semantics is None:
        return None, None
    return semantics.lower_bound, semantics.upper_bound


def metric_material_thresholds(name: str) -> tuple[float | None, float | None]:
    """Return optional per-metric material-change thresholds."""

    spec = _DEFAULT_REGISTRY.resolve(name)
    if spec is None:
        return None, None
    return spec.material_relative_delta, spec.material_absolute_delta


def _normalize_token(name: str) -> str:
    return "_".join(re.findall(r"[a-z0-9]+", name.lower()))


def _metric_tokens(name: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", name.lower()))
