import pytest

from trainlens import MonitorConfig, TrainLensMonitor


@pytest.mark.parametrize(
    ("train_name", "validation_name"),
    [
        ("train/loss", "val/loss"),
        ("train-loss", "validation-loss"),
        ("training loss", "eval loss"),
    ],
)
def test_monitor_detects_overfitting_with_normalized_loss_aliases(
    train_name: str,
    validation_name: str,
) -> None:
    monitor = TrainLensMonitor(MonitorConfig(patience=3))

    monitor.observe(1, {train_name: 1.0, validation_name: 1.0})
    monitor.observe(2, {train_name: 0.8, validation_name: 1.1})
    alerts = monitor.observe(3, {train_name: 0.6, validation_name: 1.2})

    assert any(alert.code == "possible_overfitting" for alert in alerts)
