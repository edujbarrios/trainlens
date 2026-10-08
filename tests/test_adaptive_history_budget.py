from __future__ import annotations

import pytest

from trainlens import summarize_metric_history
from trainlens.llm.context import build_llm_notebook_context
from trainlens.models.metric import MetricSeries


@pytest.mark.parametrize("epochs", [100, 200, 1000])
def test_straight_histories_use_only_endpoints(epochs: int) -> None:
    linear = tuple(1.0 - i / (epochs - 1) for i in range(epochs))
    flat = (0.45,) * epochs

    for series in (MetricSeries("loss", linear), MetricSeries("loss", flat)):
        digest = summarize_metric_history(series)
        assert digest.observations == epochs
        assert digest.sampled_positions == (1, epochs)
        assert len(digest.sampled_values) == 2


def test_context_line_gets_smaller_for_smooth_1000_epoch_history() -> None:
    values = [0.7] * 1000
    context = build_llm_notebook_context({"history": {"loss": values}})
    line = next(line for line in context.markdown.splitlines() if line.startswith("- `loss`"))
    assert "observations=1000" in line
    assert "ordered_sample=[0.7, 0.7]" in line
    assert len(line) < 140


def test_adaptive_early_exit_still_keeps_anomalies() -> None:
    values = [1.0 - i / 999 for i in range(1000)]
    values[201] += 0.1
    values[700] -= 0.15
    digest = summarize_metric_history(MetricSeries("loss", tuple(values)))

    assert 202 in digest.sampled_positions
    assert 701 in digest.sampled_positions
    assert digest.minimum_position in digest.sampled_positions
    assert digest.maximum_position in digest.sampled_positions
    assert len(digest.sampled_positions) <= 12
