from trainlens import analyze


def test_analysis_promotes_full_train_validation_test_results() -> None:
    result = analyze(
        {
            "history": {
                "train_loss": [1.1, 0.55, 0.22],
                "val_loss": [1.05, 0.61, 0.46],
                "train_accuracy": [0.64, 0.84, 0.96],
                "val_accuracy": [0.62, 0.79, 0.82],
                "val_f1": [0.60, 0.77, 0.80],
            },
            "test_metrics": {
                "loss": 0.58,
                "accuracy": 0.74,
                "f1": 0.72,
                "precision": 0.75,
                "recall": 0.70,
            },
        }
    )

    assert result.metrics["train_accuracy"] == 0.96
    assert result.metrics["validation_accuracy"] == 0.82
    assert result.metrics["validation_f1"] == 0.80
    assert result.metrics["test_accuracy"] == 0.74
    assert result.metrics["test_f1"] == 0.72
    assert result.metrics["test_precision"] == 0.75
    assert result.metrics["test_recall"] == 0.70
    assert result.metrics["test_loss"] == 0.58
    assert "Held-out test accuracy: 0.740." in result.summary
    titles = {signal.title for signal in result.signals}
    assert "Possible overfitting" in titles
    assert "Test split underperforms validation" in titles


def test_loss_only_fine_tuning_can_detect_overfitting() -> None:
    result = analyze(
        {
            "history": {
                "train_loss": [1.4, 0.8, 0.3, 0.18],
                "val_loss": [1.3, 0.9, 0.62, 0.57],
            }
        }
    )

    assert any(signal.title == "Possible overfitting" for signal in result.signals)
    assert any(
        recommendation.action.startswith("Evaluate the selected checkpoint")
        for recommendation in result.recommendations
    )
