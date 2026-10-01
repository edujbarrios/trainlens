from __future__ import annotations

from trainlens import (
    ExperimentRun,
    MetricConstraint,
    MetricSpec,
    ObjectiveSpec,
    compare_runs,
    pareto_front,
    register_metric,
    suggest_multiobjective_experiment,
    suggest_next_experiment,
    unregister_metric,
)


def test_custom_metric_registry_drives_comparison_and_planning() -> None:
    register_metric(
        MetricSpec(
            name="dice_score",
            aliases=("dice",),
            direction="higher",
            lower_bound=0.0,
            upper_bound=1.0,
            material_relative_delta=0.10,
            material_absolute_delta=0.05,
        )
    )
    try:
        comparison = compare_runs(
            {"validation_dice": 0.80},
            {"validation_dice": 0.81},
        )
        recommendation = suggest_next_experiment(
            [
                ExperimentRun(
                    name="baseline",
                    metrics={"validation_dice": 0.80},
                    parameters={"learning_rate": 1e-4},
                )
            ],
            objective_metric="validation_dice",
        )

        assert comparison.metrics[0].direction == "improved"
        assert comparison.metrics[0].magnitude == "small"
        assert recommendation.success_criteria[0].operator == ">="
    finally:
        unregister_metric("dice_score")


def test_pareto_front_preserves_quality_latency_tradeoffs() -> None:
    runs = (
        ExperimentRun(name="quality", metrics={"f1": 0.90, "latency_ms": 12.0}),
        ExperimentRun(name="fast", metrics={"f1": 0.88, "latency_ms": 8.0}),
        ExperimentRun(name="dominated", metrics={"f1": 0.85, "latency_ms": 11.0}),
    )
    objectives = (ObjectiveSpec("f1"), ObjectiveSpec("latency_ms"))

    frontier = pareto_front(runs, objectives)

    assert tuple(run.name for run in frontier) == ("quality", "fast")


def test_constraints_filter_pareto_source_and_create_multiple_criteria() -> None:
    runs = (
        ExperimentRun(
            name="quality",
            metrics={"f1": 0.90, "latency_ms": 12.0},
            parameters={"learning_rate": 1e-4},
        ),
        ExperimentRun(
            name="fast",
            metrics={"f1": 0.88, "latency_ms": 8.0},
            parameters={"learning_rate": 1e-4},
        ),
    )
    objectives = (ObjectiveSpec("f1"), ObjectiveSpec("latency_ms"))

    recommendation = suggest_multiobjective_experiment(
        runs,
        objectives,
        constraints=(MetricConstraint("latency_ms", "<=", 10.0),),
    )

    assert recommendation.source_run == "fast"
    assert tuple(item.metric for item in recommendation.success_criteria) == (
        "f1",
        "latency_ms",
    )
    assert recommendation.success_criteria[0].operator == ">="
    assert recommendation.success_criteria[1].operator == "<="
    assert any("Pareto-efficient" in item for item in recommendation.evidence)
