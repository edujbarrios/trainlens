"""Regression tests for monitoring without quadratic history snapshots."""

from trainlens import MonitorConfig, TrainLensMonitor


def test_builtin_monitor_does_not_iterate_over_full_history() -> None:
    class NoFullScan(list):
        def __iter__(self):
            raise AssertionError("built-in detectors must not scan full history")

    monitor = TrainLensMonitor(MonitorConfig(patience=3))
    monitor._observations = NoFullScan()
    for step in range(25):
        monitor.observe(step, {"train_loss": 2 - step * 0.01})
    assert len(monitor.observations) == 25


def test_custom_detectors_still_receive_complete_immutable_history() -> None:
    lengths = []

    def detector(history, current, config):
        assert isinstance(history, tuple)
        assert history[-1] is current
        lengths.append(len(history))
        return ()

    monitor = TrainLensMonitor(detectors=[detector])
    for step in range(5):
        monitor.observe(step, {"loss": 1 - 0.1 * step})
    assert lengths == [1, 2, 3, 4, 5]
