from __future__ import annotations

import pytest

from trainlens import explain_dataset


def test_explain_dataset_summarizes_columns_and_target_imbalance() -> None:
    explanation = explain_dataset(
        {
            "length": [10, 11, 12, 13, 14],
            "source": ["a", "a", "b", "a", "a"],
            "label": [1, 1, 1, 1, 0],
        },
        target="label",
    )

    split = explanation.splits[0]
    assert split.name == "dataset"
    assert split.row_count == 5
    assert [feature.name for feature in split.features] == ["length", "source"]
    assert split.features[0].kind == "numeric"
    assert split.features[0].mean == pytest.approx(12.0)
    assert split.target is not None
    assert split.target.name == "label"
    assert split.target.majority_fraction == pytest.approx(0.8)
    assert "imbalanced" in explanation.markdown
    assert "does not prove" in explanation.markdown


def test_explain_dataset_compares_split_target_distributions() -> None:
    explanation = explain_dataset(
        {
            "train": {
                "text": ["one", "two", "three", "four"],
                "label": [0, 0, 0, 1],
            },
            "validation": {
                "text": ["five", "six", "seven", "eight"],
                "label": [0, 1, 1, 1],
            },
        },
        target="label",
    )

    assert [split.name for split in explanation.splits] == ["train", "validation"]
    assert any("Target distribution differs" in item for item in explanation.observations)


def test_explain_dataset_accepts_sequence_of_rows() -> None:
    explanation = explain_dataset(
        [
            {"feature": 1.0, "label": "cat"},
            {"feature": 2.0, "label": "dog"},
            {"feature": None, "label": "cat"},
        ],
        target="label",
        name="train",
    )

    split = explanation.splits[0]
    assert split.name == "train"
    assert split.features[0].missing == 1
    assert split.target is not None
    assert split.target.unique == 2


def test_explain_dataset_accepts_hugging_face_like_columns() -> None:
    class Dataset:
        column_names = ("feature", "label")

        def __getitem__(self, name: str) -> list[object]:
            data = {
                "feature": [1, 2, 3],
                "label": ["yes", "no", "yes"],
            }
            return data[name]

    explanation = explain_dataset(Dataset(), target="label")

    assert explanation.splits[0].row_count == 3
    assert explanation.splits[0].target is not None


def test_explain_dataset_rejects_mismatched_column_lengths() -> None:
    with pytest.raises(ValueError, match="same number of rows"):
        explain_dataset({"a": [1, 2], "b": [1]})


def test_explain_dataset_limits_features_with_explicit_observation() -> None:
    explanation = explain_dataset(
        {"a": [1, 2], "b": [3, 4], "c": [5, 6]},
        max_features=2,
    )

    assert len(explanation.splits[0].features) == 2
    assert any("1 additional features" in item for item in explanation.splits[0].observations)
