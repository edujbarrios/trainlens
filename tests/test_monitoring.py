import math

import pytest

from trainlens import MonitorConfig, TrainLensMonitor


def test_monitor_emits_critical_alert_for_non_finite_metric():
    received = []
    monitor = TrainLensMonitor(on_alert=received.append)

    alerts = monitor.observe(2, {"train_loss": math.nan})

    assert alerts[0].code == "non_finite_metric"
    assert alerts[0].severity == "critical"
    assert received == list(alerts)


def test_monitor_preserves_distinct_alerts_reported_at_the_same_step():
    monitor = TrainLensMonitor()

    loss_alerts = monitor.observe(2, {"loss": math.nan})
    gradient_alerts = monitor.observe(2, {"gradient_norm": math.inf})
    duplicate_alerts = monitor.observe(2, {"gradient_norm": math.inf})

    assert loss_alerts[0].evidence == ("loss=nan",)
    assert gradient_alerts[0].evidence == ("gradient_norm=inf",)
    assert duplicate_alerts == ()


def test_monitor_detects_diverging_training_and_validation_loss():
    monitor = TrainLensMonitor(MonitorConfig(patience=3, min_delta=0.01))

    monitor.observe(1, {"train_loss": 0.8, "val_loss": 0.7})
    monitor.observe(2, {"train_loss": 0.6, "val_loss": 0.8})
    alerts = monitor.observe(3, {"train_loss": 0.4, "val_loss": 0.9})

    assert [alert.code for alert in alerts] == ["possible_overfitting"]
    assert alerts[0].evidence == ("train_loss: 0.8 -> 0.4", "val_loss: 0.7 -> 0.9")


def test_monitor_detects_stagnant_loss_with_configured_threshold():
    monitor = TrainLensMonitor(
        MonitorConfig(patience=3, min_delta=0.02, detect_overfitting=False)
    )

    monitor.observe(1, {"loss": 0.5})
    monitor.observe(2, {"loss": 0.51})
    alerts = monitor.observe(3, {"loss": 0.505})

    assert [alert.code for alert in alerts] == ["stagnation:loss"]


def test_monitor_keeps_an_immutable_observation_history():
    monitor = TrainLensMonitor()
    monitor.observe(1, {"accuracy": 1})

    assert monitor.observations[0].metrics == {"accuracy": 1.0}
    assert isinstance(monitor.observations, tuple)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"patience": 1}, "patience"), ({"min_delta": -0.1}, "min_delta")],
)
def test_monitor_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        MonitorConfig(**kwargs)


@pytest.mark.parametrize("patience", [2.5, True, "3"])
def test_monitor_rejects_non_integer_patience(patience):
    with pytest.raises(TypeError, match="patience must be an integer"):
        MonitorConfig(patience=patience)


@pytest.mark.parametrize("min_delta", [math.nan, math.inf, -math.inf])
def test_monitor_rejects_non_finite_min_delta(min_delta):
    with pytest.raises(ValueError, match="min_delta must be finite"):
        MonitorConfig(min_delta=min_delta)
