"""Plotting helpers for notebook training diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from trainlens.analyzers.metrics import MetricSplit, extract_metric_series, metric_splits
from trainlens.metric_semantics import metric_direction


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


XAxis = Literal["auto", "step", "observation"]


def plot_training_curves(
    namespace: Mapping[str, Any],
    *,
    metrics: Sequence[str] = ("loss", "accuracy"),
    splits: Sequence[MetricSplit] = ("train", "validation", "test"),
    figsize: tuple[float, float] | None = None,
    x_axis: XAxis = "auto",
    max_plot_points: int = 1200,
    show_best: bool = True,
) -> Any:
    """Visualize split-aware training curves without obscuring long histories.

    Matplotlib is part of the default installation and is loaded only when plotting. `auto`
    uses recorded steps if every non-scalar curve has a complete step axis,
    otherwise it plots *all* curves against observation indices. Explicit
    `step` mode preserves missing step metadata as gaps instead of inventing
    epoch/step coordinates. Only displayed points are downsampled; the
    underlying histories are unchanged.
    """

    if isinstance(max_plot_points, bool) or not isinstance(max_plot_points, int):
        raise TypeError("max_plot_points must be an integer")
    if max_plot_points < 4:
        raise ValueError("max_plot_points must be at least 4 to preserve extrema")
    if x_axis not in {"auto", "step", "observation"}:
        raise ValueError("x_axis must be 'auto', 'step', or 'observation'")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            'Matplotlib is missing. Reinstall with `pip install --upgrade trainlens`.'
        ) from exc

    curves = training_curves(namespace, metrics=metrics, splits=splits)
    present_metrics = tuple(metric for metric in metrics if any(c.metric == metric for c in curves))
    if not present_metrics:
        requested = ", ".join(metrics)
        raise ValueError(f"No plottable metric series found for: {requested}.")

    mode = _resolved_axis_mode(curves, x_axis)
    figure_size = figsize or (5.5 * len(present_metrics), 4.2)
    figure, axes_grid = plt.subplots(
        1, len(present_metrics), figsize=figure_size, squeeze=False,
    )
    axes = tuple(axes_grid[0])
    for axis, metric in zip(axes, present_metrics, strict=True):
        metric_curves = tuple(curve for curve in curves if curve.metric == metric)
        reference_last_x = _reference_last_x(metric_curves, mode=mode)
        for curve in metric_curves:
            all_x = _x_values(curve, mode=mode, scalar_fallback=reference_last_x)
            indices = _display_indices(curve.values, max_plot_points)
            x_values = tuple(all_x[i] for i in indices)
            y_values = tuple(curve.values[i] for i in indices)
            is_test = curve.split == "test" and len(curve.values) == 1
            partial_steps = mode == "step" and any(step is None for step in all_x)
            markers = "D" if is_test else ("." if partial_steps else "o")
            if not is_test and not partial_steps and len(indices) > 30:
                markers = ""
            axis.plot(
                x_values,
                y_values,
                marker=markers,
                markersize=7 if is_test else 3.5,
                linestyle="None" if is_test or partial_steps else "-",
                linewidth=1.6,
                label="test (held-out)" if is_test else curve.split,
            )
            if show_best and curve.split == "validation" and len(curve.values) > 1:
                direction = metric_direction(metric)
                if direction is not None:
                    best_index = (
                        min(range(len(curve.values)), key=curve.values.__getitem__)
                        if direction == "lower"
                        else max(range(len(curve.values)), key=curve.values.__getitem__)
                    )
                    best_x = all_x[best_index]
                    if best_x is not None:
                        axis.scatter(
                            [best_x],
                            [curve.values[best_index]],
                            marker="*",
                            s=110,
                            zorder=5,
                            label="best validation",
                        )
        axis.set_title(metric.replace("_", " ").title())
        axis.set_xlabel(
            "Recorded step (unknown steps omitted)" if mode == "step" else "Observation"
        )
        axis.set_ylabel(metric.replace("_", " "))
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize="small")
    figure.tight_layout()
    return figure


def _resolved_axis_mode(curves: Sequence[TrainingCurve], requested: XAxis) -> XAxis:
    histories = tuple(curve for curve in curves if len(curve.values) > 1)
    if requested == "observation":
        return "observation"
    if requested == "step":
        if any(not curve.steps for curve in histories):
            raise ValueError(
                "x_axis='step' requires recorded steps for every non-scalar curve"
            )
        return "step"
    if histories and all(
        curve.steps and all(step is not None for step in curve.steps) for curve in histories
    ):
        return "step"
    return "observation"


def _reference_last_x(
    curves: Sequence[TrainingCurve], *, mode: XAxis = "observation"
) -> int | float | None:
    candidates: list[int | float] = []
    for curve in curves:
        if curve.split == "test":
            continue
        x_values = _x_values(curve, mode=mode)
        candidates.extend(value for value in x_values if value is not None)
    return max(candidates) if candidates else None


def _x_values(
    curve: TrainingCurve,
    *,
    mode: XAxis = "observation",
    scalar_fallback: int | float | None = None,
) -> tuple[int | float | None, ...]:
    if mode == "step" and curve.steps:
        # Unknown positions stay unknown; do not reinterpret valid recorded steps.
        return curve.steps
    if len(curve.values) == 1 and scalar_fallback is not None:
        return (scalar_fallback,)
    return tuple(range(1, len(curve.values) + 1))


def _display_indices(values: tuple[float, ...], limit: int) -> tuple[int, ...]:
    """Bound plotted points using min/max buckets to retain sharp excursions.

    Unlike a uniform stride, each bucket contributes both extrema. Memory is
    bounded by the display budget, and the work is linear in source length.
    """

    length = len(values)
    if length <= limit:
        return tuple(range(length))
    buckets = (limit - 2) // 2
    chosen = {0, length - 1}
    interior = length - 2
    for bucket in range(buckets):
        start = 1 + bucket * interior // buckets
        end = 1 + (bucket + 1) * interior // buckets
        if start == end:
            continue
        chosen.add(min(range(start, end), key=values.__getitem__))
        chosen.add(max(range(start, end), key=values.__getitem__))
    return tuple(sorted(chosen))
