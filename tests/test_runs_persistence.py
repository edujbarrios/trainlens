from __future__ import annotations

from datetime import UTC, datetime

from trainlens import TrainingRun, compare_runs, load_run, render_run_comparison, save_run
from trainlens.models.metric import MetricSeries


def test_training_run_json_roundtrip_preserves_history_and_configuration(tmp_path) -> None:
    run = TrainingRun(
        run_id="exp-1",
        created_at=datetime(2026, 10, 1, 18, 0, tzinfo=UTC),
        model_name="demo",
        framework="pytorch",
        metrics=(
            MetricSeries(
                name="validation_loss",
                values=(0.8, 0.5, 0.4),
                steps=(1, 2, 3),
                split="validation",
            ),
        ),
        parameters={"learning_rate": 0.001, "batch_size": 32},
        metadata={"dataset": "demo-v1", "seed": 42},
        notes=("baseline",),
    )

    path = save_run(run, tmp_path / "run.json")
    restored = load_run(path)

    assert restored == run


def test_compare_training_runs_reports_configuration_and_best_trajectory() -> None:
    baseline = TrainingRun(
        run_id="baseline",
        metrics=(MetricSeries("validation_loss", (0.8, 0.5, 0.45)),),
        parameters={"learning_rate": 0.001, "dropout": 0.1},
    )
    experiment = TrainingRun(
        run_id="experiment",
        metrics=(MetricSeries("validation_loss", (0.75, 0.42, 0.40)),),
        parameters={"learning_rate": 0.0005, "dropout": 0.1},
    )

    comparison = compare_runs(baseline, experiment)
    rendered = render_run_comparison(comparison)

    assert comparison.parameter_changes[0].name == "learning_rate"
    assert comparison.trajectories[0].baseline_best == 0.45
    assert comparison.trajectories[0].experiment_best == 0.40
    assert comparison.trajectories[0].baseline_observations == 3
    assert "### Configuration changes" in rendered
    assert "### Training trajectories" in rendered
