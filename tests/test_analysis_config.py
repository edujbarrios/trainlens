from __future__ import annotations

import pytest

from trainlens import AnalysisConfig, analyze, preview_notebook_context


class DummyModel:
    def fit(self) -> None:
        pass

    def predict(self) -> None:
        pass


class DummyTrainer:
    def __init__(self, model: object, learning_rate: float) -> None:
        self.model = model
        self.args = type("Args", (), {"learning_rate": learning_rate})()


def test_strict_analysis_rejects_ambiguous_models() -> None:
    namespace = {"first": DummyModel(), "second": DummyModel()}

    with pytest.raises(ValueError, match="multiple model candidates"):
        analyze(namespace, config=AnalysisConfig(strict=True))


def test_explicit_model_selection_is_reflected_in_analysis() -> None:
    first = DummyModel()
    second = DummyModel()

    result = analyze(
        {"first": first, "second": second},
        config=AnalysisConfig(model="second"),
    )

    assert any("`second`" in item for item in result.summary)


def test_explicit_metrics_override_are_analyzed() -> None:
    model = DummyModel()
    result = analyze(
        {"model": model},
        config=AnalysisConfig(
            model="model",
            metrics={
                "train_loss": [0.8, 0.5, 0.3],
                "validation_loss": [0.9, 0.7, 0.8],
            },
        ),
    )

    assert result.metrics["train_loss"] == pytest.approx(0.3)
    assert result.metrics["validation_loss"] == pytest.approx(0.8)


def test_preview_context_uses_explicit_model() -> None:
    first = DummyModel()
    second = DummyModel()

    context = preview_notebook_context(
        {"first": first, "second": second},
        analysis_config=AnalysisConfig(model="second"),
    )

    selected_lines = [line for line in context.markdown.splitlines() if "selected" in line]
    assert len(selected_lines) == 1
    assert "`second`" in selected_lines[0]
