from trainlens import MonitorConfig, TrainingAlert, TrainLensMonitor


def test_custom_detector_runs_on_first_observation():
    seen = []

    def detector(history, current, config):
        seen.append((history, current, config))
        value = current.metrics.get("gradient_norm")
        if value is None or value <= 10:
            return ()
        return (
            TrainingAlert(
                code="high_gradient_norm",
                severity="warning",
                message="Gradient norm exceeded the configured policy.",
                evidence=(f"gradient_norm={value}",),
                step=current.step,
            ),
        )

    config = MonitorConfig(patience=4)
    monitor = TrainLensMonitor(config, detectors=(detector,))

    alerts = monitor.observe(1, {"gradient_norm": 12.0})

    assert [alert.code for alert in alerts] == ["high_gradient_norm"]
    assert seen[0][0] == monitor.observations
    assert seen[0][1] == monitor.observations[-1]
    assert seen[0][2] is config
    assert monitor.detectors == (detector,)


def test_custom_persistent_detector_rearms_after_condition_clears():
    def detector(history, current, config):
        del history, config
        value = current.metrics.get("learning_rate")
        if value is None or value <= 0.1:
            return ()
        return (
            TrainingAlert(
                code="learning_rate_too_high",
                severity="warning",
                message="Learning rate is above the project policy.",
                evidence=(f"learning_rate={value}",),
                step=current.step,
                persistent=True,
            ),
        )

    monitor = TrainLensMonitor()
    monitor.add_detector(detector)

    first = monitor.observe(1, {"learning_rate": 0.2})
    repeated = monitor.observe(2, {"learning_rate": 0.3})
    cleared = monitor.observe(3, {"learning_rate": 0.05})
    recurring = monitor.observe(4, {"learning_rate": 0.2})

    assert [alert.code for alert in first] == ["learning_rate_too_high"]
    assert repeated == ()
    assert cleared == ()
    assert [alert.code for alert in recurring] == ["learning_rate_too_high"]
