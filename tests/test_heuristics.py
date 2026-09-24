from trainlens.heuristics import detect_class_imbalance, detect_overfitting
from trainlens.models.metric import MetricSeries


def test_detects_overfitting_gap():
    signal = detect_overfitting(
        MetricSeries("train_accuracy", (0.7, 0.95)),
        MetricSeries("validation_accuracy", (0.68, 0.78)),
    )

    assert signal is not None
    assert signal.severity == "warning"


def test_detects_class_imbalance():
    signal = detect_class_imbalance([0] * 95 + [1] * 5)

    assert signal is not None
    assert "smallest class" in signal.detail


def test_balanced_multiclass_dataset_is_not_flagged() -> None:
    labels = [label for label in range(10) for _ in range(100)]

    assert detect_class_imbalance(labels) is None


def test_detects_strong_multiclass_imbalance() -> None:
    signal = detect_class_imbalance([0] * 80 + [1] * 10 + [2] * 10)

    assert signal is not None
    assert "3 classes" in signal.detail
