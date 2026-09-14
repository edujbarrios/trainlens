import math

from trainlens.models.analysis import AnalysisResult, Recommendation, Signal
from trainlens.renderers.markdown import MarkdownRenderer


def test_markdown_renderer_includes_recommendations():
    result = AnalysisResult(
        model_name="RandomForestClassifier",
        framework="sklearn",
        metrics={"validation_accuracy": 0.82, "train_accuracy": 0.91},
        signals=[Signal("Possible overfitting", "Gap detected", "warning")],
        recommendations=[Recommendation("Tune max_depth.", "Reduce memorization.")],
    )

    markdown = MarkdownRenderer().render(result)

    assert "RandomForestClassifier" in markdown
    assert "| validation_accuracy | 0.820 |" in markdown
    assert "### Result explanation" in markdown
    assert "Tune max_depth" in markdown
    assert "### Improvement plan" in markdown


def test_markdown_renderer_explains_loss_gap_and_trace():
    result = AnalysisResult(
        metrics={"train_loss": 1.2, "validation_loss": 1.7},
        recommendations=[Recommendation("Lower learning rate.", "Validation loss is lagging.")],
    )

    markdown = MarkdownRenderer().render(result)

    assert "Validation loss is materially higher" in markdown
    assert "Lower learning rate" in markdown


def test_markdown_renderer_escapes_metric_table_cells():
    markdown = MarkdownRenderer().render(AnalysisResult(metrics={"precision|recall\nmacro": 0.5}))

    assert r"| precision\|recall macro | 0.500 |" in markdown


def test_markdown_renderer_does_not_interpret_non_finite_loss_gap():
    markdown = MarkdownRenderer().render(
        AnalysisResult(metrics={"train_loss": math.nan, "validation_loss": 1.0})
    )

    assert "loss is materially" not in markdown
    assert "loss are close" not in markdown
