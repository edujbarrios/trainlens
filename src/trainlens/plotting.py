"""Optional plotting helpers for notebook training diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from trainlens.analyzers.metrics import MetricSplit, extract_metric_series, metric_splits


@dataclass(frozen=True)
class TrainingCurve:
    """One plot-ready train/validation/test metric series."""

    metric: str
    split: MetricSplit
    values: tuple[float, ...]
    steps: tuple[int | float | None, ...] = ()


def training_curves(
    namespace: Mapping[str, Any],
    *,
    metrics: Sequence[str] = ("loss", "accuracy"),
    splits: Sequence[MetricSplit] = ("train", "validation", "test"),
) -> tuple[TrainingCurve, ...]:
    """Return normalized plot-ready curves for common fine-tuning splits."""

    series = extract_metric_series(namespace)
    curves: list[TrainingCurve] = []
    for metric in metrics:
        by_split = metric_splits(series, metric)
        for split in splits:
            item = by_split.get(split)
            if item is None or not item.values:
                continue
            curves.append(
                TrainingCurve(
                    metric=metric,
                    split=split,
                    values=item.values,
                    steps=item.steps,
                )
            )
    return tuple(curves)


def plot_training_curves(
    namespace: Mapping[str, Any],
    *,
    metrics: Sequence[str] = ("loss", "accuracy"),
    splits: Sequence[MetricSplit] = ("train", "validation", "test"),
    figsize: tuple[float, float] | None = None,
) -> Any:
    """Plot train/validation/test curves with an optional Matplotlib dependency.

    Install ``trainlens[plots]`` to enable this helper. The function returns the
    Matplotlib ``Figure`` so callers can further customize or save it.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            'Plotting requires Matplotlib. Install it with `pip install "trainlens[plots]"`.'
        ) from exc

    curves = training_curves(namespace, metrics=metrics, splits=splits)
    present_metrics = tuple(metric for metric in metrics if any(c.metric == metric for c in curves))
    if not present_metrics:
        requested = ", ".join(metrics)
        raise ValueError(f"No plottable metric series found for: {requested}.")

    width = 5.5 * len(present_metrics)
    figure_size = figsize or (width, 4.0)
    figure, axes_grid = plt.subplots(
        1,
        len(present_metrics),
        figsize=figure_size,
        squeeze=False,
    )
    axes = tuple(axes_grid[0])
    for axis, metric in zip(axes, present_metrics, strict=True):
        metric_curves = tuple(curve for curve in curves if curve.metric == metric)
        reference_last_x = _reference_last_x(metric_curves)
        for curve in metric_curves:
            x_values = _x_values(curve, scalar_fallback=reference_last_x)
            axis.plot(x_values, curve.values, marker="o", label=curve.split)
        axis.set_title(metric.replace("_", " ").title())
        axis.set_xlabel("step / epoch")
        axis.set_ylabel(metric.replace("_", " "))
        axis.grid(True, alpha=0.25)
        axis.legend()
    figure.tight_layout()
    return figure


def _reference_last_x(curves: Sequence[TrainingCurve]) -> int | float | None:
    candidates: list[int | float] = []
    for curve in curves:
        if curve.split == "test":
            continue
        x_values = _x_values(curve)
        if x_values:
            candidates.append(x_values[-1])
    return max(candidates) if candidates else None


def _x_values(
    curve: TrainingCurve,
    *,
    scalar_fallback: int | float | None = None,
) -> tuple[int | float, ...]:
    if curve.steps and all(step is not None for step in curve.steps):
        return tuple(step for step in curve.steps if step is not None)
    if len(curve.values) == 1 and scalar_fallback is not None:
        return (scalar_fallback,)
    return tuple(range(1, len(curve.values) + 1))
