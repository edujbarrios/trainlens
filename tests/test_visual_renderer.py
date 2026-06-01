import shutil
from pathlib import Path

from trainlens.models.analysis import AnalysisResult, Recommendation, Signal
from trainlens.models.trace import TraceEvent
from trainlens.renderers.visual import DarkVisualRenderer


def test_visual_renderer_outputs_dark_metric_svg():
    result = AnalysisResult(
        metrics={"validation_loss": 1.2},
        trace=[
            TraceEvent(step=1, metrics={"loss": 2.0}),
            TraceEvent(step=2, metrics={"loss": 1.5, "eval_loss": 1.8}),
        ],
    )

    svg = DarkVisualRenderer().render_metric_trace(result)

    assert svg.startswith('<svg xmlns="http://www.w3.org/2000/svg"')
    assert 'fill="#0b1020"' in svg
    assert "<title id=\"trainlens-metric-trace-title\">TrainLens metric trace</title>" in svg
    assert "Metric Trace" in svg
    assert "eval_loss" in svg


def test_visual_renderer_outputs_overview_card():
    result = AnalysisResult(
        model_name="LlavaProjector",
        framework="transformers",
        metrics={"validation_loss": 1.2},
        trace=[TraceEvent(step=1, metrics={"loss": 2.0})],
    )

    svg = DarkVisualRenderer().render_overview_card(result)

    assert "TrainLens Overview" in svg
    assert "LlavaProjector" in svg
    assert "Trace events" in svg


def test_visual_renderer_outputs_signal_and_feature_views():
    result = AnalysisResult(
        signals=[Signal("Possible overfitting", "Validation gap detected.", "warning")],
        top_features=["caption_length", "projector_lr"],
    )
    renderer = DarkVisualRenderer()

    assert "Possible overfitting" in renderer.render_signal_panel(result)
    assert "caption_length" in renderer.render_feature_lens(result)


def test_visual_renderer_outputs_trace_timeline():
    result = AnalysisResult(
        trace=[
            TraceEvent(step=10, name="batch_end", metrics={"loss": 2.0}),
            TraceEvent(step=20, name="eval", metrics={"eval_loss": 1.8}),
            TraceEvent(step=30, name="checkpoint", message="saved adapter"),
        ]
    )

    svg = DarkVisualRenderer().render_trace_timeline(result)

    assert "Execution Timeline" in svg
    assert "batch_end" in svg
    assert "eval_loss=1.8" in svg


def test_visual_renderer_writes_all_dashboard_assets():
    result = AnalysisResult(trace=[TraceEvent(step=1, name="batch_end", metrics={"loss": 2.0})])
    output_dir = Path(".trainlens") / "visual-renderer-test-assets"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    assets = DarkVisualRenderer().write_dashboard_assets(result, output_dir)

    assert sorted(assets) == [
        "feature_lens",
        "improvement_plan",
        "metric_trace",
        "overview",
        "signal_panel",
        "trace_timeline",
    ]
    assert assets["trace_timeline"].exists()

    shutil.rmtree(output_dir)


def test_visual_renderer_outputs_inline_dashboard_html():
    result = AnalysisResult(trace=[TraceEvent(step=1, name="batch_end", metrics={"loss": 2.0})])

    html = DarkVisualRenderer().render_dashboard_html(result)

    assert 'class="trainlens-dark-dashboard"' in html
    assert html.count("<svg ") == 6
    assert "Execution Timeline" in html


def test_visual_renderer_outputs_improvement_plan():
    result = AnalysisResult(
        recommendations=[
            Recommendation("Lower learning rate.", "Validation loss is lagging.", 0.82),
        ]
    )

    svg = DarkVisualRenderer().render_improvement_plan(result)

    assert "Improvement Plan" in svg
    assert "Lower learning rate" in svg
    assert "82%" in svg
