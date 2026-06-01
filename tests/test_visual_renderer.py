from trainlens.models.analysis import AnalysisResult, Signal
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
