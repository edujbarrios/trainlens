from __future__ import annotations

from trainlens import explain_dataset, preview_notebook_context
from trainlens.llm.verified import evidence_catalog
from trainlens.models.analysis import AnalysisResult


def test_dataset_explanation_can_be_added_to_notebook_context_without_raw_rows() -> None:
    dataset = {
        "text": [
            "private training row alpha",
            "private training row beta",
            "private training row gamma",
            "private training row delta",
        ],
        "label": [1, 1, 1, 0],
    }
    explanation = explain_dataset(dataset, target="label", name="train")

    context = preview_notebook_context(
        {"history": {"loss": [1.0, 0.8], "val_loss": [1.1, 0.9]}},
        dataset_explanation=explanation,
    )

    assert "## Dataset Context" in context.markdown
    assert "TrainLens Dataset Explanation" in context.markdown
    assert "Target `label` is imbalanced" in context.markdown
    assert "private training row alpha" not in context.markdown
    assert "private training row beta" not in context.markdown


def test_dataset_evidence_receives_stable_citable_ids() -> None:
    explanation = explain_dataset(
        {
            "train": {
                "length": [10, 20, 30, 40],
                "label": [0, 0, 0, 1],
            },
            "validation": {
                "length": [12, 18, 31, 45],
                "label": [0, 1, 1, 1],
            },
        },
        target="label",
    )

    catalog = evidence_catalog(AnalysisResult(), dataset_explanation=explanation)
    ids = {item.evidence_id for item in catalog}

    assert any(item.startswith("dataset:split:train-") and item.endswith(":rows") for item in ids)
    assert any(":feature:length-" in item for item in ids)
    assert any(item.startswith("dataset:split:train-") and ":target:label-" in item for item in ids)
    assert any(
        item.startswith("dataset:split:validation-") and ":target:label-" in item
        for item in ids
    )
    assert any(item.startswith("dataset:observation:") for item in ids)


def test_dataset_evidence_contains_aggregate_values_not_feature_rows() -> None:
    explanation = explain_dataset(
        {
            "text": ["secret sample one", "secret sample two"],
            "label": ["yes", "no"],
        },
        target="label",
    )

    catalog = evidence_catalog(AnalysisResult(), dataset_explanation=explanation)
    combined = "\n".join(item.detail for item in catalog)

    assert "feature=text" in combined
    assert "average_length=" in combined
    assert "secret sample one" not in combined
    assert "secret sample two" not in combined
