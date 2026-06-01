"""Dark-mode SVG visual explanations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from html import escape
from pathlib import Path

from trainlens.models.analysis import AnalysisResult, Signal
from trainlens.models.trace import TraceEvent

_BACKGROUND = "#0b1020"
_PANEL = "#111827"
_PANEL_ALT = "#172033"
_GRID = "#293246"
_TEXT = "#e5edf6"
_MUTED = "#94a3b8"
_CYAN = "#22d3ee"
_GREEN = "#34d399"
_AMBER = "#f59e0b"
_RED = "#fb7185"
_PURPLE = "#a78bfa"
_BLUE = "#60a5fa"


class DarkVisualRenderer:
    """Render notebook explanations as dependency-free dark-mode SVG images."""

    def render_metric_trace(self, result: AnalysisResult) -> str:
        series = _series_from_trace(result.trace)
        if not series and result.metrics:
            series = {
                name: [(index + 1, value)]
                for index, (name, value) in enumerate(result.metrics.items())
            }
        width, height = 920, 420
        plot_x, plot_y, plot_w, plot_h = 78, 76, 770, 260
        lines = _svg_header(width, height, "TrainLens metric trace")
        lines.extend(
            [
                _text(42, 46, "Metric Trace", 26, weight=700),
                _text(42, 70, "Recent training signals rendered in dark mode", 13, _MUTED),
                _rect(plot_x, plot_y, plot_w, plot_h, _PANEL_ALT, 10, opacity=0.84),
            ]
        )
        for offset in range(5):
            y = plot_y + offset * plot_h / 4
            lines.append(
                f'<line x1="{plot_x}" y1="{y:.1f}" '
                f'x2="{plot_x + plot_w}" y2="{y:.1f}" '
                f'stroke="{_GRID}" stroke-width="1" />'
            )
        if not series:
            lines.append(_text(plot_x + 26, plot_y + 142, "No trace metrics found yet", 18, _MUTED))
            return _svg_footer(lines)

        all_points = [point for points in series.values() for point in points]
        min_step = min(step for step, _value in all_points)
        max_step = max(step for step, _value in all_points)
        min_value = min(value for _step, value in all_points)
        max_value = max(value for _step, value in all_points)
        if min_value == max_value:
            min_value -= 1
            max_value += 1
        colors = (_CYAN, _GREEN, _AMBER, _PURPLE, _BLUE)
        for index, (name, points) in enumerate(series.items()):
            color = colors[index % len(colors)]
            coordinates = [
                (
                    _scale(step, min_step, max_step, plot_x, plot_x + plot_w),
                    _scale(value, min_value, max_value, plot_y + plot_h, plot_y),
                )
                for step, value in points
            ]
            polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in coordinates)
            lines.append(
                f'<polyline points="{polyline}" fill="none" stroke="{color}" '
                'stroke-width="4" stroke-linecap="round" stroke-linejoin="round" />'
            )
            for x, y in coordinates:
                lines.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" '
                    f'fill="{color}" stroke="{_BACKGROUND}" stroke-width="2" />'
                )
            legend_y = 104 + index * 30
            lines.append(_rect(868, legend_y - 12, 14, 14, color, 3))
            lines.append(_text(890, legend_y, name, 13, _TEXT))
        lines.append(_text(plot_x, plot_y + plot_h + 34, f"step {min_step}", 12, _MUTED))
        lines.append(
            _text(plot_x + plot_w - 60, plot_y + plot_h + 34, f"step {max_step}", 12, _MUTED)
        )
        lines.append(_text(26, plot_y + 12, f"{max_value:.3g}", 12, _MUTED))
        lines.append(_text(26, plot_y + plot_h, f"{min_value:.3g}", 12, _MUTED))
        return _svg_footer(lines)

    def render_signal_panel(self, result: AnalysisResult) -> str:
        width, height = 920, 420
        lines = _svg_header(width, height, "TrainLens signal panel")
        lines.extend(
            [
                _text(42, 46, "Signal Panel", 26, weight=700),
                _text(
                    42,
                    70,
                    "Training risks and useful findings, ordered for quick inspection",
                    13,
                    _MUTED,
                ),
            ]
        )
        signals = result.signals or [
            Signal(
                "No critical signal detected",
                "TrainLens did not find high-confidence risks.",
                "info",
            )
        ]
        for index, signal in enumerate(signals[:5]):
            y = 104 + index * 58
            color = _severity_color(signal.severity)
            strength = {"info": 0.42, "warning": 0.68, "critical": 0.92}[signal.severity]
            lines.append(_rect(42, y, 830, 42, _PANEL_ALT, 9, opacity=0.9))
            lines.append(_rect(42, y, 830 * strength, 42, color, 9, opacity=0.2))
            lines.append(_rect(42, y, 8, 42, color, 4))
            lines.append(_text(66, y + 18, signal.title, 15, _TEXT, 700))
            lines.append(_text(66, y + 35, signal.detail, 12, _MUTED))
            lines.append(_text(806, y + 26, signal.severity.upper(), 12, color, 700))
        return _svg_footer(lines)

    def render_feature_lens(self, result: AnalysisResult) -> str:
        width, height = 920, 420
        lines = _svg_header(width, height, "TrainLens feature lens")
        lines.extend(
            [
                _text(42, 46, "Feature Lens", 26, weight=700),
                _text(42, 70, "SHAP-inspired ranked feature explanation, always dark", 13, _MUTED),
            ]
        )
        features = result.top_features[:7] or [
            "validation_loss",
            "train_loss",
            "lora_rank",
            "dataset_balance",
        ]
        max_width = 620
        for index, name in enumerate(features):
            y = 112 + index * 38
            impact = (len(features) - index) / len(features)
            bar_width = max_width * impact
            color = (_CYAN, _GREEN, _AMBER, _PURPLE, _BLUE)[index % 5]
            lines.append(_text(42, y + 15, f"{index + 1}", 13, _MUTED, 700))
            lines.append(_text(76, y + 15, name, 14, _TEXT))
            lines.append(_rect(250, y, max_width, 20, _PANEL_ALT, 10))
            lines.append(_rect(250, y, bar_width, 20, color, 10, opacity=0.86))
            lines.append(_text(250 + bar_width + 14, y + 15, f"{impact:.2f}", 12, _MUTED))
        return _svg_footer(lines)

    def write_dashboard_assets(
        self, result: AnalysisResult, directory: Path | str
    ) -> dict[str, Path]:
        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        assets = {
            "metric_trace": output_dir / "trainlens-dark-metric-trace.svg",
            "signal_panel": output_dir / "trainlens-dark-signal-panel.svg",
            "feature_lens": output_dir / "trainlens-dark-feature-lens.svg",
        }
        assets["metric_trace"].write_text(self.render_metric_trace(result), encoding="utf-8")
        assets["signal_panel"].write_text(self.render_signal_panel(result), encoding="utf-8")
        assets["feature_lens"].write_text(self.render_feature_lens(result), encoding="utf-8")
        return assets


def _series_from_trace(trace: list[TraceEvent]) -> Mapping[str, list[tuple[int, float]]]:
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for index, event in enumerate(trace, start=1):
        step = event.step or index
        for name, value in event.metrics.items():
            grouped[name].append((step, value))
    return {name: points for name, points in grouped.items() if points}


def _svg_header(width: int, height: int, label: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(label)}">',
        f'<rect width="{width}" height="{height}" fill="{_BACKGROUND}" />',
        f'<rect x="18" y="18" width="{width - 36}" height="{height - 36}" '
        f'rx="18" fill="{_PANEL}" stroke="{_GRID}" />',
    ]


def _svg_footer(lines: list[str]) -> str:
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _text(
    x: float,
    y: float,
    value: str,
    size: int,
    color: str = _TEXT,
    weight: int = 500,
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" fill="{color}" '
        'font-family="Inter, Segoe UI, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}">{escape(value)}</text>'
    )


def _rect(
    x: float,
    y: float,
    width: float,
    height: float,
    fill: str,
    radius: float,
    opacity: float = 1,
) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'rx="{radius:.1f}" fill="{fill}" opacity="{opacity:.2f}" />'
    )


def _scale(
    value: float,
    source_min: float,
    source_max: float,
    target_min: float,
    target_max: float,
) -> float:
    if source_min == source_max:
        return (target_min + target_max) / 2
    ratio = (value - source_min) / (source_max - source_min)
    return target_min + ratio * (target_max - target_min)


def _severity_color(severity: str) -> str:
    if severity == "critical":
        return _RED
    if severity == "warning":
        return _AMBER
    return _CYAN
