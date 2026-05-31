from trainlens.analyzers.metrics import extract_metric_series, paired_metric


def test_extracts_keras_style_history():
    series = extract_metric_series(
        {"history": {"accuracy": [0.7, 0.8], "val_accuracy": [0.68, 0.75]}}
    )

    train, validation = paired_metric(series, "accuracy")

    assert train is not None
    assert validation is not None
    assert train.last == 0.8
    assert validation.split == "validation"


def test_extracts_trainer_style_log_history():
    series = extract_metric_series(
        {
            "log_history": [
                {"step": 0, "loss": 2.4},
                {"step": 1, "loss": 2.1, "eval_loss": 2.3},
                {"step": 2, "loss": 1.9, "eval_loss": 2.0},
            ]
        }
    )

    train, validation = paired_metric(series, "loss")

    assert train is not None
    assert validation is not None
    assert train.values == (2.4, 2.1, 1.9)
    assert train.steps == (0, 1, 2)
    assert validation.values == (2.3, 2.0)
    assert validation.steps == (1, 2)
