from __future__ import annotations

import pytest

from trainlens.history import summarize_metric_history
from trainlens.llm.context import build_llm_notebook_context
from trainlens.models.metric import MetricSeries


@pytest.mark.parametrize("length", [100, 200, 1000])
def test_history_digest_is_bounded_and_keeps_extrema(length: int) -> None:
    values = [0.9 - index * 0.0003 for index in range(length)]
    values[length // 3] = 5.0
    values[2 * length // 3] = -4.0
    series = MetricSeries("loss", tuple(values), steps=tuple(range(1, length + 1)))

    digest = summarize_metric_history(series, max_points=12)

    assert digest.observations == length
    assert len(digest.sampled_positions) <= 12
    assert digest.sampled_positions[0] == 1
    assert digest.sampled_positions[-1] == length
    assert digest.minimum == -4.0
    assert digest.maximum == 5.0
    assert digest.minimum_position in digest.sampled_positions
    assert digest.maximum_position in digest.sampled_positions
    assert digest.sampled_steps == digest.sampled_positions


def test_digest_captures_local_excursion_even_when_it_is_not_global_extreme() -> None:
    values = [1.0 - 0.0005 * index for index in range(1000)]
    values[501] += 0.12
    digest = summarize_metric_history(MetricSeries("loss", tuple(values)), max_points=5)

    assert 502 in digest.sampled_positions


def test_digest_preserves_unknown_steps_without_fabricating_steps() -> None:
    values = [1.0] * 100
    values[63] = 2.0
    steps = tuple(None if i == 63 else i * 10 for i in range(100))
    digest = summarize_metric_history(MetricSeries("loss", tuple(values), steps=steps))

    assert None in digest.sampled_steps
    assert digest.sampled_steps[digest.sampled_positions.index(64)] is None


def test_digest_has_small_budget_and_input_validation() -> None:
    series = MetricSeries("loss", (1.0, 0.8, 0.5))
    assert len(summarize_metric_history(series, max_points=2).sampled_values) == 2
    with pytest.raises(ValueError, match="at least 2"):
        summarize_metric_history(series, max_points=1)
    with pytest.raises(TypeError, match="integer"):
        summarize_metric_history(series, max_points=True)
    with pytest.raises(ValueError, match="empty"):
        summarize_metric_history(MetricSeries("loss", ()))
    with pytest.raises(ValueError, match="finite"):
        summarize_metric_history(MetricSeries("loss", (float("nan"), 0.5)))


def test_context_keeps_anomalies_and_bounds_long_epochs() -> None:
    values = [1.0 - i * 0.0005 for i in range(1000)]
    values[237] = 6.123
    values[814] = -2.345
    context = build_llm_notebook_context({"history": {"loss": values}})

    line = next(line for line in context.markdown.splitlines() if line.startswith("- `loss`"))
    assert "observations=1000" in line
    assert "max=6.123" in line
    assert "min=-2.345" in line
    assert "max_at=238" in line
    assert "min_at=815" in line
    assert "6.123" in line.split("ordered_sample=")[1]
    assert "-2.345" in line.split("ordered_sample=")[1]
    assert len(line) < 500
    assert len(values) == 1000
