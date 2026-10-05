from __future__ import annotations

import pytest

from trainlens import MetricConstraint, ObjectiveSpec, TrainingRun, leaderboard
from trainlens.models.metric import MetricSeries


def _run(run_id: str, loss: float, latency: float, quality: float = 0.0) -> TrainingRun:
    return TrainingRun(
        run_id=run_id,
        metrics=(
            MetricSeries("validation_loss", (loss,)),
            MetricSeries("latency_ms", (latency,)),
            MetricSeries("quality", (quality,)),
        ),
    )


def test_leaderboard_ranks_primary_objective_and_marks_pareto_runs() -> None:
    board = leaderboard(
        (
            _run("baseline", 0.50, 8.0),
            _run("better-loss", 0.40, 9.0),
            _run("too-slow", 0.35, 12.0),
        ),
        (ObjectiveSpec("validation_loss"), ObjectiveSpec("latency_ms")),
        constraints=(MetricConstraint("latency_ms", "<=", 10.0),),
    )

    assert [row.run_id for row in board.rows] == ["better-loss", "baseline", "too-slow"]
    assert [row.rank for row in board.rows] == [1, 2, None]
    assert board.rows[0].pareto
    assert board.rows[1].pareto
    assert not board.rows[2].satisfies_constraints
    assert "| 1 | better-loss | 0.4 | 9 | yes | yes |" in board.to_markdown()
    assert board._repr_markdown_() == board.to_markdown()


def test_leaderboard_supports_explicit_direction_for_custom_metrics() -> None:
    board = leaderboard(
        (_run("low", 0.5, 8.0, 1.0), _run("high", 0.6, 8.0, 2.0)),
        (ObjectiveSpec("quality", direction="max"),),
    )

    assert board.rows[0].run_id == "high"
    assert board.rows[0].rank == 1


def test_leaderboard_validates_objectives_ids_and_unknown_direction() -> None:
    run = _run("same", 0.5, 8.0)

    with pytest.raises(ValueError, match="at least one objective"):
        leaderboard((run,), ())
    with pytest.raises(ValueError, match="unique run_id"):
        leaderboard((run, run), (ObjectiveSpec("validation_loss"),))
    with pytest.raises(ValueError, match="cannot infer"):
        leaderboard((run,), (ObjectiveSpec("mystery_metric"),))
