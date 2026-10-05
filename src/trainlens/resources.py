"""Portable resource and cost metrics for training runs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite

from trainlens.models.metric import MetricSeries
from trainlens.models.run import TrainingRun


@dataclass(frozen=True)
class ResourceProfile:
    """Measured resource values that can participate in TrainLens objectives."""

    duration_seconds: float
    items_processed: int | None = None
    tokens_processed: int | None = None
    peak_memory_mb: float | None = None
    gpu_memory_mb: float | None = None
    estimated_cost_usd: float | None = None

    @property
    def throughput_per_second(self) -> float | None:
        if self.items_processed is None:
            return None
        return self.items_processed / self.duration_seconds

    @property
    def token_throughput_per_second(self) -> float | None:
        if self.tokens_processed is None:
            return None
        return self.tokens_processed / self.duration_seconds

    def metrics(self) -> dict[str, float]:
        """Return resource values as ordinary scalar metrics."""

        output = {"duration_seconds": self.duration_seconds}
        if self.throughput_per_second is not None:
            output["throughput_per_second"] = self.throughput_per_second
        if self.token_throughput_per_second is not None:
            output["token_throughput_per_second"] = self.token_throughput_per_second
        if self.peak_memory_mb is not None:
            output["peak_memory_mb"] = self.peak_memory_mb
        if self.gpu_memory_mb is not None:
            output["gpu_memory_mb"] = self.gpu_memory_mb
        if self.estimated_cost_usd is not None:
            output["estimated_cost_usd"] = self.estimated_cost_usd
        return output


def profile_resources(
    *,
    duration_seconds: float,
    items_processed: int | None = None,
    tokens_processed: int | None = None,
    peak_memory_mb: float | None = None,
    gpu_memory_mb: float | None = None,
    estimated_cost_usd: float | None = None,
) -> ResourceProfile:
    """Build a validated resource profile from measurements supplied by the caller."""

    duration = _positive_float(duration_seconds, "duration_seconds")
    items = _non_negative_int(items_processed, "items_processed")
    tokens = _non_negative_int(tokens_processed, "tokens_processed")
    peak = _optional_non_negative_float(peak_memory_mb, "peak_memory_mb")
    gpu = _optional_non_negative_float(gpu_memory_mb, "gpu_memory_mb")
    cost = _optional_non_negative_float(estimated_cost_usd, "estimated_cost_usd")
    return ResourceProfile(
        duration_seconds=duration,
        items_processed=items,
        tokens_processed=tokens,
        peak_memory_mb=peak,
        gpu_memory_mb=gpu,
        estimated_cost_usd=cost,
    )


def attach_resource_profile(run: TrainingRun, profile: ResourceProfile) -> TrainingRun:
    """Attach resource values to a portable run as final scalar metric series."""

    resource_metrics = profile.metrics()
    existing = {series.name: series for series in run.metrics}
    for name, value in resource_metrics.items():
        existing[name] = MetricSeries(name=name, values=(value,))
    return replace(run, metrics=tuple(existing[name] for name in sorted(existing)))


def _positive_float(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{label} must be numeric")
    numeric = float(value)
    if not isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return numeric


def _optional_non_negative_float(value: float | None, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{label} must be numeric or null")
    numeric = float(value)
    if not isfinite(numeric) or numeric < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return numeric


def _non_negative_int(value: int | None, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer or null")
    if value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value
