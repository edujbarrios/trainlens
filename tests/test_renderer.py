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
    assert "Tune max_depth" in markdown
