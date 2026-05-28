from trainlens.analyzers.metrics import extract_metric_series, paired_metric


def test_extracts_keras_style_history():
    series = extract_metric_series({"history": {"accuracy": [0.7, 0.8], "val_accuracy": [0.68, 0.75]}})

    train, validation = paired_metric(series, "accuracy")

    assert train is not None
    assert validation is not None
    assert train.last == 0.8
    assert validation.split == "validation"
