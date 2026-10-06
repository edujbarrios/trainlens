from __future__ import annotations

import pytest

from trainlens import TrainingRun, aggregate_runs, compare_run_groups, parameter_effects
from trainlens.models.metric import MetricSeries


def _run(
    run_id: str,
    *,
    learning_rate: float,
    dropout: float,
    loss: float,
    seed: int,
) -> TrainingRun:
    return TrainingRun(
        run_id=run_id,
        metrics=(MetricSeries("validation_loss", (loss,)),),
        parameters={"learning_rate": learning_rate, "dropout": dropout},
        metadata={"seed": seed},
    )


def test_aggregate_runs_summarizes_repeated_configurations() -> None:
    runs = (
        _run("a1", learning_rate=0.001, dropout=0.1, loss=0.50, seed=1),
        _run("a2", learning_rate=0.001, dropout=0.1, loss=0.54, seed=2),
        _run("b1", learning_rate=0.0005, dropout=0.1, loss=0.43, seed=1),
        _run("b2", learning_rate=0.0005, dropout=0.1, loss=0.45, seed=2),
    )

    groups = aggregate_runs(runs, by=("learning_rate", "dropout"))

    assert len(groups) == 2
    assert groups[0].metrics["validation_loss"].count == 2
    assert groups[0].metrics["validation_loss"].mean == pytest.approx(0.44)
    assert groups[1].metrics["validation_loss"].mean == pytest.approx(0.52)


def test_compare_run_groups_reports_effect_relative_to_seed_noise() -> None:
    baseline_runs = tuple(
        _run(
            f"a{seed}",
            learning_rate=0.001,
            dropout=0.1,
            loss=value,
            seed=seed,
        )
        for seed, value in enumerate((0.50, 0.52, 0.48), start=1)
    )
    candidate_runs = tuple(
        _run(
            f"b{seed}",
            learning_rate=0.0005,
            dropout=0.1,
            loss=value,
            seed=seed,
        )
        for seed, value in enumerate((0.40, 0.42, 0.38), start=1)
    )
    groups = aggregate_runs((*baseline_runs, *candidate_runs), by=("learning_rate",))

    comparison = compare_run_groups(groups[1], groups[0], metric="validation_loss")

    assert comparison.delta == pytest.approx(-0.10)
    assert comparison.improvement == pytest.approx(0.10)
    assert comparison.signal_to_noise is not None
    assert comparison.signal_to_noise > 1.0
    assert comparison.confidence == "medium"


def test_parameter_effects_use_only_single_parameter_controlled_pairs() -> None:
    runs = (
        _run("a", learning_rate=0.001, dropout=0.1, loss=0.50, seed=1),
        _run("b", learning_rate=0.0005, dropout=0.1, loss=0.40, seed=1),
        _run("c", learning_rate=0.001, dropout=0.2, loss=0.48, seed=1),
        _run("d", learning_rate=0.0005, dropout=0.2, loss=0.39, seed=1),
    )

    effects = parameter_effects(runs, metric="validation_loss")
    lr_effect = next(item for item in effects if item.parameter == "learning_rate")

    assert lr_effect.pairs == 2
    assert lr_effect.from_value == 0.0005
    assert lr_effect.to_value == 0.001
    assert lr_effect.median_delta == pytest.approx(0.095)
    assert lr_effect.median_improvement == pytest.approx(-0.095)
    assert lr_effect.improvement_rate == 0.0
    assert len(lr_effect.evidence) == 2


def test_repeated_run_helpers_validate_inputs_and_unknown_metrics() -> None:
    run = _run("a", learning_rate=0.001, dropout=0.1, loss=0.5, seed=1)

    with pytest.raises(ValueError, match="grouping parameter"):
        aggregate_runs((run,), by=())

    group = aggregate_runs((run,), by=("learning_rate",))[0]
    with pytest.raises(ValueError, match="must exist"):
        compare_run_groups(group, group, metric="accuracy")
    with pytest.raises(ValueError, match="cannot infer"):
        parameter_effects((run,), metric="custom")


def test_parameter_effects_match_repeated_runs_one_to_one_by_seed() -> None:
    runs = tuple(
        _run(
            f"{prefix}{seed}",
            learning_rate=learning_rate,
            dropout=0.1,
            loss=losses[seed - 1],
            seed=seed,
        )
        for prefix, learning_rate, losses in (
            ("a", 0.001, (0.50, 0.52, 0.48)),
            ("b", 0.0005, (0.40, 0.42, 0.38)),
        )
        for seed in range(1, 4)
    )

    effects = parameter_effects(runs, metric="validation_loss")
    lr_effect = next(item for item in effects if item.parameter == "learning_rate")

    assert lr_effect.pairs == 3
    assert len(lr_effect.evidence) == 3
    assert all("seed=" in item for item in lr_effect.evidence)
