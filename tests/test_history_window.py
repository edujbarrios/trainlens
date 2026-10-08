from __future__ import annotations

import pytest

from trainlens import HistoryDigest, inspect_history_window, summarize_metric_history
from trainlens.models.metric import MetricSeries


def test_public_digest_and_window_preserve_original_observation_positions() -> None:
    values = [1.0] * 1000
    values[550] = -9.0
    series = MetricSeries(
        "loss", tuple(values), split="validation", steps=tuple(range(5, 10005, 10))
    )

    entire = summarize_metric_history(series, max_points=8)
    window = inspect_history_window(series, start=540, end=570, max_points=8)

    assert isinstance(entire, HistoryDigest)
    assert entire.minimum_position == 551
    assert window.observations == 31
    assert window.minimum_position == 551
    assert 551 in window.sampled_positions
    assert window.sampled_steps[window.sampled_positions.index(551)] == 5505
    assert window.sampled_positions[0] == 540
    assert window.sampled_positions[-1] == 570
    assert len(series.values) == 1000
    assert series.values[550] == -9.0


@pytest.mark.parametrize(
    ("start", "end"),
    [(0, 10), (11, 10), (1, 1001), (1002, None)],
)
def test_window_rejects_invalid_ranges(start: int, end: int | None) -> None:
    series = MetricSeries("loss", tuple(range(1000)))
    with pytest.raises(ValueError, match="window"):
        inspect_history_window(series, start=start, end=end)


def test_window_does_not_invent_missing_steps() -> None:
    series = MetricSeries("loss", (3.0, 1.0, 4.0), steps=(1, None, 9))
    window = inspect_history_window(series, start=2, end=3, max_points=2)
    assert window.sampled_positions == (2, 3)
    assert window.sampled_steps == (None, 9)


@pytest.mark.parametrize("invalid", [True, 1.5, "3"])
def test_window_rejects_non_integer_positions(invalid: object) -> None:
    series = MetricSeries("loss", (1.0, 2.0, 3.0))
    with pytest.raises(TypeError):
        inspect_history_window(series, start=invalid)  # type: ignore[arg-type]
