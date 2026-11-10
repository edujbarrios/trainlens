"""Scale-aware deterministic diagnostics for a single metric trajectory."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from statistics import median

from trainlens.models.metric import MetricSeries


@dataclass(frozen=True)
class TrendDiagnostic:
    """A cautious assessment of the recent trajectory, not a causal explanation."""

    metric: str
    state: str
    recent_change: float
    relative_change: float
    noise_floor: float
    observations: int
    explanation: str


def diagnose_trend(
    series: MetricSeries,
    *,
    window: int = 5,
    relative_tolerance: float = 0.01,
    noise_multiplier: float = 1.5,
) -> TrendDiagnostic | None:
    """Compare recent change with signal scale and typical historical fluctuations.

    Returns None for insufficient or non-finite observations. Absolute changes
    alone are never used to infer convergence. A recent trend can be stable,
    improving, or worsening; direction must be registered for the latter two.
    """
    if isinstance(window, bool) or not isinstance(window, int) or window < 3:
        raise ValueError("window must be an integer >= 3")
    if not isfinite(relative_tolerance) or relative_tolerance < 0:
        raise ValueError("relative_tolerance must be finite and non-negative")
    if not isfinite(noise_multiplier) or noise_multiplier < 0:
        raise ValueError("noise_multiplier must be finite and non-negative")
    values = series.values
    if len(values) < window or not all(isfinite(v) for v in values):
        return None
    tail = values[-window:]
    changes = [right - left for left, right in zip(values, values[1:], strict=False)]
    # A robust noise estimate based on deviation from the median increment.
    baseline = median(changes)
    noise = median(abs(value - baseline) for value in changes) * 1.4826
    scale = max(abs(median(tail)), max(tail) - min(tail), 1e-12)
    change = tail[-1] - tail[0]
    relative = abs(change) / scale
    tolerance = max(relative_tolerance * scale, noise_multiplier * noise)
    if abs(change) <= tolerance:
        state = "stable"
        explanation = "Recent change is small relative to metric scale and observed noise."
    else:
        from trainlens.metric_semantics import metric_direction

        direction = metric_direction(series.name)
        if direction is None:
            state = "changing"
            explanation = "A recent change is detected, but metric direction is unknown."
        else:
            better = change > 0 if direction == "higher" else change < 0
            state = "improving" if better else "worsening"
            explanation = "Recent change exceeds the scale-aware noise threshold."
    return TrendDiagnostic(
        metric=series.name,
        state=state,
        recent_change=change,
        relative_change=relative,
        noise_floor=tolerance,
        observations=window,
        explanation=explanation,
    )
