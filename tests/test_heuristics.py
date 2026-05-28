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
