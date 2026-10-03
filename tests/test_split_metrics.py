from trainlens.analyzers.metrics import extract_metric_series, metric_splits


def test_metric_containers_preserve_train_validation_and_test_splits() -> None:
    series = extract_metric_series(
        {
            "train_metrics": {"accuracy": 0.94, "f1": 0.93},
            "validation_metrics": {"accuracy": 0.88, "f1": 0.87},
            "test_metrics": {"accuracy": 0.84, "f1": 0.82},
        }
    )

    assert series["train_accuracy"].last == 0.94
    assert series["train_accuracy"].split == "train"
    assert series["validation_f1"].last == 0.87
    assert series["validation_f1"].split == "validation"
    assert series["test_accuracy"].last == 0.84
    assert series["test_accuracy"].split == "test"
    assert series["test_f1"].last == 0.82


def test_test_prefixes_and_holdout_aliases_normalize_to_test_split() -> None:
    series = extract_metric_series(
        {
            "test_loss": 0.42,
            "testing_accuracy": 0.86,
            "holdout_metrics": {"perplexity": 8.4},
        }
    )

    assert series["test_loss"].split == "test"
    assert series["test_accuracy"].last == 0.86
    assert series["test_perplexity"].last == 8.4


def test_metric_splits_returns_full_fine_tuning_evaluation_view() -> None:
    series = extract_metric_series(
        {
            "history": {
                "train_loss": [1.2, 0.7, 0.4],
                "val_loss": [1.1, 0.72, 0.55],
            },
            "test_metrics": {"loss": 0.61},
        }
    )

    splits = metric_splits(series, "loss")

    assert tuple(splits) == ("train", "validation", "test")
    assert splits["train"].last == 0.4
    assert splits["validation"].last == 0.55
    assert splits["test"].last == 0.61
