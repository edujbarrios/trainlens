from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

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


def test_training_run_consumers_keep_metric_splits_distinct() -> None:
    baseline = TrainingRun(
        run_id="baseline-splits",
        metrics=(
            MetricSeries("loss", (0.20,), split="train"),
            MetricSeries("loss", (0.30,), split="validation"),
            MetricSeries("loss", (0.40,), split="test"),
        ),
    )
    experiment = TrainingRun(
        run_id="experiment-splits",
        metrics=(
            MetricSeries("loss", (0.18,), split="train"),
            MetricSeries("loss", (0.28,), split="validation"),
            MetricSeries("loss", (0.45,), split="test"),
        ),
    )

    comparison = compare_runs(baseline, experiment)
    by_name = {item.name: item for item in comparison.metrics}

    assert set(by_name) == {"train_loss", "validation_loss", "test_loss"}
    assert by_name["validation_loss"].baseline == pytest.approx(0.30)
    assert by_name["test_loss"].experiment == pytest.approx(0.45)


def test_portable_run_json_rejects_boolean_and_non_finite_metric_values(tmp_path) -> None:
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "bad",
                "created_at": datetime.now(UTC).isoformat(),
                "model_name": None,
                "framework": None,
                "parameters": {},
                "metadata": {},
                "notes": [],
                "metrics": [{"name": "loss", "values": [True], "split": None, "steps": []}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="finite numbers"):
        load_run(invalid_path)

    non_finite = TrainingRun(
        run_id="nan",
        metrics=(MetricSeries("loss", (float("nan"),)),),
    )
    with pytest.raises(ValueError):
        save_run(non_finite, tmp_path / "nan.json")
