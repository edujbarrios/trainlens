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


def test_empty_columns_are_not_misclassified_as_dataset_splits() -> None:
    explanation = explain_dataset({"feature": [], "label": []}, target="label")

    assert len(explanation.splits) == 1
    assert explanation.splits[0].name == "dataset"
    assert explanation.splits[0].row_count == 0
    assert [feature.name for feature in explanation.splits[0].features] == ["feature"]


def test_non_finite_numeric_values_do_not_contaminate_statistics() -> None:
    explanation = explain_dataset({"feature": [1.0, float("inf"), 3.0]})

    feature = explanation.splits[0].features[0]
    assert feature.minimum == 1.0
    assert feature.maximum == 3.0
    assert feature.mean == 2.0


def test_optional_pandas_missing_value_is_counted_when_available() -> None:
    pandas = pytest.importorskip("pandas")
    explanation = explain_dataset({"feature": [1.0, pandas.NA, 3.0]})

    assert explanation.splits[0].features[0].missing == 1


def test_bounded_profile_only_consumes_requested_rows() -> None:
    from collections.abc import Sequence

    class LargeColumn(Sequence):
        def __len__(self) -> int:
            return 1_000_000

        def __getitem__(self, index):
            if isinstance(index, slice):
                raise AssertionError("unbounded slicing is forbidden")
            if index >= 3:
                raise AssertionError("requested too many rows")
            return index + 1

    explanation = explain_dataset({"value": LargeColumn()}, max_rows=3)
    assert explanation.splits[0].row_count == 3
    assert explanation.splits[0].features[0].mean == 2.0
    assert "at most the first 3 rows" in explanation.splits[0].observations[0]


def test_bounded_profile_validates_positive_row_budget() -> None:
    with pytest.raises(ValueError, match="max_rows"):
        explain_dataset({"value": [1]}, max_rows=0)
