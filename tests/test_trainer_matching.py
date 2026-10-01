from trainlens.llm.context import build_llm_notebook_context_from_snapshot
from trainlens.models.snapshot import FrameworkArtifact, NotebookSnapshot
from trainlens.pipeline import analyze_snapshot


class FakeArgs:
    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate


class FakeTrainer:
    def __init__(self, model: object, learning_rate: float) -> None:
        self.model = model
        self.args = FakeArgs(learning_rate)


def _multi_trainer_snapshot() -> NotebookSnapshot:
    model_a = object()
    model_b = object()
    trainer_a = FakeTrainer(model_a, 1e-4)
    trainer_b = FakeTrainer(model_b, 9e-4)
    return NotebookSnapshot(
        variables=(),
        framework_artifacts=(
            FrameworkArtifact(
                variable_name="trainer_b",
                framework="huggingface",
                type_name="FakeTrainer",
                history={},
                model_name="ModelB",
                model_ref=model_b,
                confidence=0.60,
            ),
            FrameworkArtifact(
                variable_name="trainer_a",
                framework="huggingface",
                type_name="FakeTrainer",
                history={},
                model_name="ModelA",
                model_ref=model_a,
                confidence=0.90,
            ),
        ),
        raw_namespace={"trainer_b": trainer_b, "trainer_a": trainer_a},
    )


def test_analysis_uses_trainer_for_selected_model() -> None:
    result = analyze_snapshot(_multi_trainer_snapshot())

    assert any(line == "Learning rates: base=0.0001." for line in result.summary)
    assert all("base=0.0009" not in line for line in result.summary)


def test_llm_context_uses_trainer_for_selected_model() -> None:
    context = build_llm_notebook_context_from_snapshot(_multi_trainer_snapshot())

    assert "`training.learning_rate`: 0.0001" in context.markdown
    assert "`training.learning_rate`: 0.0009" not in context.markdown
