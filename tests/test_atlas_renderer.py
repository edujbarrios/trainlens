from trainlens.models.analysis import AnalysisResult, Recommendation, Signal
from trainlens.models.trace import TraceEvent
from trainlens.renderers.atlas import AtlasRenderer


def test_atlas_renderer_outputs_notebook_ui():
    result = AnalysisResult(
        model_name="MistralLoRA",
        framework="transformers",
        metrics={"train_loss": 1.2, "validation_loss": 1.5},
        signals=[Signal("Validation gap", "Validation loss is lagging.", "warning")],
        recommendations=[
            Recommendation("Inspect long-answer validation examples.", "Dataset note."),
        ],
        trace=[TraceEvent(step=10, name="eval", metrics={"eval_loss": 1.5})],
    )

    html = AtlasRenderer().render(result)

    assert 'class="trainlens-atlas"' in html
    assert "MistralLoRA" in html
    assert "validation_loss" in html
    assert "Training signal detected" in html
