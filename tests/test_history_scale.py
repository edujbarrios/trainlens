"""Regressions for the scale of anomaly-aware history selection."""

from trainlens import summarize_metric_history
from trainlens.models.metric import MetricSeries


def test_large_baseline_does_not_hide_local_training_spike() -> None:
    values = [1_000_000.0 - i * 0.01 for i in range(1000)]
    values[383] += 3.0  # small relative to baseline, huge relative to range
    values[617] += 2.0  # meaningful local deviation, not a global extremum

    digest = summarize_metric_history(MetricSeries("loss", tuple(values)), max_points=12)

    assert 384 in digest.sampled_positions
    assert 618 in digest.sampled_positions
    assert len(digest.sampled_positions) <= 12


def test_large_offset_linear_history_still_uses_just_endpoints() -> None:
    series = MetricSeries("loss", tuple(10_000.0 + i * 0.25 for i in range(1000)))
    digest = summarize_metric_history(series)

    assert digest.sampled_positions == (1, 1000)


def test_tiny_scale_spike_remains_visible() -> None:
    values = [1e-9 - i * 1e-13 for i in range(1000)]
    values[444] += 2e-10
    digest = summarize_metric_history(MetricSeries("loss", tuple(values)), max_points=8)

    assert 445 in digest.sampled_positions
