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

    def render_dashboard_html(self, result: AnalysisResult) -> str:
        visuals = [
            self.render_overview_card(result),
            self.render_metric_trace(result),
            self.render_signal_panel(result),
            self.render_trace_timeline(result),
            self.render_feature_lens(result),
            self.render_improvement_plan(result),
        ]
        cards = "\n".join(
            f'<section class="trainlens-visual-card">{visual}</section>' for visual in visuals
        )
        return (
            '<div class="trainlens-dark-dashboard">'
            "<style>"
            ".trainlens-dark-dashboard{background:#0b1020;color:#e5edf6;"
            "display:grid;gap:18px;padding:18px;border-radius:14px;"
            "grid-template-columns:repeat(auto-fit,minmax(340px,1fr));}"
            ".trainlens-dark-dashboard svg{width:100%;height:auto;display:block;}"
            ".trainlens-visual-card{min-width:0;}"
            "</style>"
            f"{cards}"
            "</div>"
        )

    def render_overview_card(self, result: AnalysisResult) -> str:
        width, height = 920, 420
        model = result.model_name or "No confident model detected"
        framework = result.framework or "framework unknown"
        lines = _svg_header(
            width,
            height,
            "TrainLens overview",
            "Summary card with model, framework, metrics, signals, trace events, and next steps.",
        )
        lines.extend(
            [
                _text(42, 46, "TrainLens Overview", 26, weight=700),
                _text(42, 70, "Dark visual summary of the current training run", 13, _MUTED),
                _text(42, 126, model, 30, _TEXT, 800),
                _text(42, 154, framework, 15, _MUTED),
            ]
        )
        stat_cards = (
            ("Signals", len(result.signals), _severity_color(_highest_severity(result.signals))),
            ("Metrics", len(result.metrics), _CYAN),
            ("Trace events", len(result.trace), _GREEN),
            ("Next steps", len(result.recommendations), _PURPLE),
        )
        for index, (label, stat_value, color) in enumerate(stat_cards):
            x = 42 + index * 210
            lines.append(_rect(x, 198, 182, 86, _PANEL_ALT, 12, opacity=0.9))
            lines.append(_rect(x, 198, 182, 4, color, 2))
            lines.append(_text(x + 18, 234, str(stat_value), 30, _TEXT, 800))
            lines.append(_text(x + 18, 260, label, 13, _MUTED, 600))
        metric_items = sorted(result.metrics.items())[:4]
        if metric_items:
            lines.append(_text(42, 326, "Key metrics", 15, _TEXT, 700))
            for index, (name, metric_value) in enumerate(metric_items):
                x = 42 + index * 210
                lines.append(_rect(x, 346, 182, 34, _BACKGROUND, 8, opacity=0.7))
                lines.append(
                    _text(x + 14, 368, f"{name} {metric_value:.3f}", 12, _MUTED, 600)
                )
        else:
            lines.append(_text(42, 344, "No scalar metrics found yet", 15, _MUTED, 600))
        return _svg_footer(lines)

    def render_metric_trace(self, result: AnalysisResult) -> str:
        series = _series_from_trace(result.trace)
        if not series and result.metrics:
            series = {
                name: [(index + 1, value)]
                for index, (name, value) in enumerate(result.metrics.items())
            }
        width, height = 920, 420
        plot_x, plot_y, plot_w, plot_h = 78, 76, 770, 260
        lines = _svg_header(
            width,
            height,
            "TrainLens metric trace",
            "Line chart showing numeric training metrics collected from execution trace events.",
        )
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
        lines = _svg_header(
            width,
            height,
            "TrainLens signal panel",
            "Severity panel showing training risks and useful findings.",
        )
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

    def render_trace_timeline(self, result: AnalysisResult) -> str:
        width, height = 920, 420
        lines = _svg_header(
            width,
            height,
            "TrainLens execution timeline",
            "Timeline showing recent training events, checkpoints, and evaluations.",
        )
        lines.extend(
            [
                _text(42, 46, "Execution Timeline", 26, weight=700),
                _text(42, 70, "Recent run events connected to training evidence", 13, _MUTED),
            ]
        )
        events = result.trace[-7:]
        if not events:
            lines.append(_text(42, 210, "No execution events found yet", 18, _MUTED))
            return _svg_footer(lines)

        axis_x, axis_y, axis_w = 82, 196, 756
        lines.append(
            f'<line x1="{axis_x}" y1="{axis_y}" x2="{axis_x + axis_w}" y2="{axis_y}" '
            f'stroke="{_GRID}" stroke-width="2" />'
        )
        for index, event in enumerate(events):
            x = axis_x + (axis_w * index / max(len(events) - 1, 1))
            color = _event_color(event)
            label = _event_label(event)
            metric_label = _compact_metrics(event.metrics)
            lines.append(_rect(x - 8, axis_y - 8, 16, 16, color, 8))
            lines.append(_text(x - 28, axis_y - 28, _event_position(event, index), 12, _MUTED, 700))
            card_y = 238 if index % 2 == 0 else 104
            lines.append(_rect(x - 66, card_y, 132, 62, _PANEL_ALT, 9, opacity=0.92))
            lines.append(_rect(x - 66, card_y, 132, 4, color, 2))
            lines.append(_text(x - 54, card_y + 24, _clip(label, 17), 13, _TEXT, 700))
            lines.append(_text(x - 54, card_y + 44, _clip(metric_label, 20), 11, _MUTED, 600))
            connector_y = card_y if card_y > axis_y else card_y + 62
            lines.append(
                f'<line x1="{x:.1f}" y1="{axis_y}" x2="{x:.1f}" y2="{connector_y}" '
                f'stroke="{color}" stroke-width="1.5" opacity="0.65" />'
            )
        return _svg_footer(lines)

    def render_feature_lens(self, result: AnalysisResult) -> str:
        width, height = 920, 420
        lines = _svg_header(
            width,
            height,
            "TrainLens feature lens",
            "Ranked evidence bars for model features and training signals.",
        )
        lines.extend(
            [
                _text(42, 46, "Feature Lens", 26, weight=700),
                _text(
                    42,
                    70,
                    "Ranked training evidence for quick dark-mode inspection",
                    13,
                    _MUTED,
                ),
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

    def render_improvement_plan(self, result: AnalysisResult) -> str:
        width, height = 920, 420
        lines = _svg_header(
            width,
            height,
            "TrainLens improvement plan",
            "Prioritized recommended actions with confidence and rationale.",
        )
        lines.extend(
            [
                _text(42, 46, "Improvement Plan", 26, weight=700),
                _text(42, 70, "Prioritized next experiments from the report", 13, _MUTED),
            ]
        )
        recommendations = sorted(
            result.recommendations, key=lambda item: item.confidence, reverse=True
        )[:4]
        if not recommendations:
            lines.append(_text(42, 210, "No recommendations generated yet", 18, _MUTED))
            return _svg_footer(lines)
        for index, recommendation in enumerate(recommendations):
            y = 104 + index * 72
            confidence_width = 170 * max(min(recommendation.confidence, 1), 0)
            color = (_GREEN, _CYAN, _AMBER, _PURPLE)[index % 4]
            lines.append(_rect(42, y, 830, 54, _PANEL_ALT, 10, opacity=0.92))
            lines.append(_rect(42, y, 8, 54, color, 4))
            lines.append(_text(66, y + 21, _clip(recommendation.action, 68), 14, _TEXT, 700))
            lines.append(_text(66, y + 41, _clip(recommendation.rationale, 86), 11, _MUTED))
            lines.append(_rect(682, y + 17, 170, 10, _BACKGROUND, 5, opacity=0.8))
            lines.append(_rect(682, y + 17, confidence_width, 10, color, 5, opacity=0.9))
            lines.append(
                _text(682, y + 43, f"{int(recommendation.confidence * 100)}%", 11, color, 700)
            )
        return _svg_footer(lines)

    def write_dashboard_assets(
        self, result: AnalysisResult, directory: Path | str
    ) -> dict[str, Path]:
        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        assets = {
            "overview": output_dir / "trainlens-dark-overview.svg",
            "metric_trace": output_dir / "trainlens-dark-metric-trace.svg",
            "signal_panel": output_dir / "trainlens-dark-signal-panel.svg",
            "trace_timeline": output_dir / "trainlens-dark-trace-timeline.svg",
            "feature_lens": output_dir / "trainlens-dark-feature-lens.svg",
            "improvement_plan": output_dir / "trainlens-dark-improvement-plan.svg",
        }
        assets["overview"].write_text(self.render_overview_card(result), encoding="utf-8")
        assets["metric_trace"].write_text(self.render_metric_trace(result), encoding="utf-8")
        assets["signal_panel"].write_text(self.render_signal_panel(result), encoding="utf-8")
        assets["trace_timeline"].write_text(self.render_trace_timeline(result), encoding="utf-8")
        assets["feature_lens"].write_text(self.render_feature_lens(result), encoding="utf-8")
        assets["improvement_plan"].write_text(
            self.render_improvement_plan(result), encoding="utf-8"
        )
        return assets


def _series_from_trace(trace: list[TraceEvent]) -> Mapping[str, list[tuple[int, float]]]:
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for index, event in enumerate(trace, start=1):
        step = event.step or index
        for name, value in event.metrics.items():
            grouped[name].append((step, value))
    return {name: points for name, points in grouped.items() if points}


def _svg_header(width: int, height: int, label: str, description: str) -> list[str]:
    title_id = _slug_id(label, "title")
    description_id = _slug_id(label, "description")
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        f'aria-labelledby="{title_id} {description_id}">',
        f'<title id="{title_id}">{escape(label)}</title>',
        f'<desc id="{description_id}">{escape(description)}</desc>',
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


def _event_color(event: TraceEvent) -> str:
    label = ((event.name or "") + " " + (event.message or "")).lower()
    if "checkpoint" in label or "save" in label:
        return _PURPLE
    has_validation_metric = any(
        name.startswith(("eval_", "val_", "validation_")) for name in event.metrics
    )
    if "eval" in label or has_validation_metric:
        return _AMBER
    if "error" in label or "fail" in label:
        return _RED
    return _CYAN


def _event_label(event: TraceEvent) -> str:
    return event.name or event.message or "training event"


def _event_position(event: TraceEvent, index: int) -> str:
    if event.step is not None:
        return f"step {event.step}"
    if event.epoch is not None:
        return f"epoch {event.epoch:g}"
    return f"event {index + 1}"


def _compact_metrics(metrics: Mapping[str, float]) -> str:
    if not metrics:
        return "no scalar metrics"
    name, value = next(iter(metrics.items()))
    return f"{name}={value:.3g}"


def _clip(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(limit - 1, 0)] + "..."


def _highest_severity(signals: list[Signal]) -> str:
    severities = {signal.severity for signal in signals}
    if "critical" in severities:
        return "critical"
    if "warning" in severities:
        return "warning"
    return "info"


def _slug_id(value: str, suffix: str) -> str:
    normalized = "".join(character.lower() if character.isalnum() else "-" for character in value)
    compact = "-".join(part for part in normalized.split("-") if part)
    return f"{compact}-{suffix}"
