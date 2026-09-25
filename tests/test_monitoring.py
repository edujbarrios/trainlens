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


def test_distinct_non_finite_events_remain_observable_across_advancing_steps():
    monitor = TrainLensMonitor()

    first = monitor.observe(1, {"loss": math.nan})
    second = monitor.observe(2, {"loss": math.nan})

    assert [alert.code for alert in first] == ["non_finite_metric"]
    assert [alert.code for alert in second] == ["non_finite_metric"]
    assert first[0].step == 1
    assert second[0].step == 2


def test_monitor_detects_diverging_training_and_validation_loss():
    monitor = TrainLensMonitor(MonitorConfig(patience=3, min_delta=0.01))

    monitor.observe(1, {"train_loss": 0.8, "val_loss": 0.7})
    monitor.observe(2, {"train_loss": 0.6, "val_loss": 0.8})
    alerts = monitor.observe(3, {"train_loss": 0.4, "val_loss": 0.9})

    assert [alert.code for alert in alerts] == ["possible_overfitting"]
    assert alerts[0].evidence == ("train_loss: 0.8 -> 0.4", "val_loss: 0.7 -> 0.9")


def test_persistent_overfitting_alert_is_suppressed_until_condition_clears():
    monitor = TrainLensMonitor(MonitorConfig(patience=3, min_delta=0.01))

    monitor.observe(1, {"train_loss": 0.8, "val_loss": 0.7})
    monitor.observe(2, {"train_loss": 0.6, "val_loss": 0.8})
    first = monitor.observe(3, {"train_loss": 0.4, "val_loss": 0.9})
    repeated = monitor.observe(4, {"train_loss": 0.2, "val_loss": 1.0})
    cleared = monitor.observe(5, {"train_loss": 0.19, "val_loss": 0.85})
    monitor.observe(6, {"train_loss": 0.16, "val_loss": 0.9})
    recurring = monitor.observe(7, {"train_loss": 0.13, "val_loss": 0.95})

    assert [alert.code for alert in first] == ["possible_overfitting"]
    assert repeated == ()
    assert all(alert.code != "possible_overfitting" for alert in cleared)
    assert [alert.code for alert in recurring] == ["possible_overfitting"]


def test_monitor_detects_stagnant_loss_with_configured_threshold():
    monitor = TrainLensMonitor(
        MonitorConfig(patience=3, min_delta=0.02, detect_overfitting=False)
    )

    monitor.observe(1, {"loss": 0.5})
    monitor.observe(2, {"loss": 0.51})
    alerts = monitor.observe(3, {"loss": 0.505})

    assert [alert.code for alert in alerts] == ["stagnation:loss"]


def test_stagnation_alert_rearms_after_condition_clears():
    monitor = TrainLensMonitor(
        MonitorConfig(patience=3, min_delta=0.01, detect_overfitting=False)
    )

    monitor.observe(1, {"loss": 0.5})
    monitor.observe(2, {"loss": 0.5})
    first = monitor.observe(3, {"loss": 0.5})
    repeated = monitor.observe(4, {"loss": 0.5})
    cleared = monitor.observe(5, {"loss": 0.4})
    monitor.observe(6, {"loss": 0.4})
    recurring = monitor.observe(7, {"loss": 0.4})

    assert [alert.code for alert in first] == ["stagnation:loss"]
    assert repeated == ()
    assert cleared == ()
    assert [alert.code for alert in recurring] == ["stagnation:loss"]


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


@pytest.mark.parametrize("step", [True, 1.5, "1"])
def test_monitor_rejects_non_integer_steps(step):
    with pytest.raises(TypeError, match="step must be an integer"):
        TrainLensMonitor().observe(step, {"loss": 1.0})


@pytest.mark.parametrize("value", [True, False, "1.0", None])
def test_monitor_rejects_boolean_and_non_numeric_metrics(value):
    with pytest.raises(TypeError, match="must be a number"):
        TrainLensMonitor().observe(1, {"loss": value})


def test_monitor_rejects_steps_that_move_backwards():
    monitor = TrainLensMonitor()
    monitor.observe(3, {"loss": 1.0})

    with pytest.raises(ValueError, match="cannot move backwards"):
        monitor.observe(2, {"loss": 0.8})
