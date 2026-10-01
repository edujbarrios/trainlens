from __future__ import annotations

from trainlens.introspection import NotebookInspector
from trainlens.llm.context import build_llm_notebook_context_from_snapshot
from trainlens.models.analysis import AnalysisResult, EvidenceRef, Recommendation, Signal


def test_llm_context_includes_deterministic_findings() -> None:
    result = AnalysisResult(model_name="demo")
    result.summary.append("Validation loss improved.")
    result.metrics["validation_loss"] = 0.42
    result.signals.append(
        Signal(
            title="Possible overfitting",
            detail="Validation loss diverged from training loss.",
            severity="warning",
            evidence=("train_loss 0.3 -> 0.2", "validation_loss 0.4 -> 0.5"),
            evidence_refs=(
                EvidenceRef(
                    source="trainer.log_history",
                    metric="validation_loss",
                    start_step=100,
                    end_step=200,
                    detail="validation loss increased",
                ),
            ),
        )
    )
    result.recommendations.append(
        Recommendation(
            action="Increase regularization.",
            rationale="The validation gap widened.",
            confidence=0.78,
            evidence=("validation gap widened",),
            source="possible_overfitting",
        )
    )
    snapshot = NotebookInspector().snapshot({"history": {"validation_loss": [0.4, 0.5]}})

    context = build_llm_notebook_context_from_snapshot(
        snapshot,
        deterministic_result=result,
    )

    assert "## TrainLens Deterministic Findings" in context.markdown
    assert "Validation loss improved." in context.markdown
    assert "Possible overfitting" in context.markdown
    assert "trainer.log_history" in context.markdown
    assert "Increase regularization." in context.markdown
    assert "source=possible_overfitting" in context.markdown
