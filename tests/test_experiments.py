import pytest

from trainlens import ExperimentRun, suggest_next_experiment


def test_suggest_next_experiment_targets_generalization_gap():
    recommendation = suggest_next_experiment(
        [
            ExperimentRun(
                name="baseline",
                metrics={"train_loss": 0.2, "validation_loss": 0.5},
                parameters={"dropout": 0.1, "batch_size": 32},
                estimated_cost="medium",
            )
        ]
    )

    assert recommendation.source_run == "baseline"
    assert recommendation.changes == {"dropout": 0.15}
    assert recommendation.keep_constant == ("batch_size",)
    assert recommendation.success_criteria[0].metric == "validation_loss"
    assert recommendation.success_criteria[0].operator == "<="
    assert recommendation.success_criteria[0].target == pytest.approx(0.495)
    assert recommendation.estimated_cost == "medium"
    assert recommendation.evidence


def test_suggestion_uses_best_run_and_changes_one_variable():
    recommendation = suggest_next_experiment(
        [
            ExperimentRun(
                name="run-a",
                metrics={"validation_accuracy": 0.81},
                parameters={"learning_rate": 1e-3, "batch_size": 16},
            ),
            ExperimentRun(
                name="run-b",
                metrics={"validation_accuracy": 0.86},
                parameters={"learning_rate": 5e-4, "batch_size": 16},
            ),
        ]
    )

    assert recommendation.source_run == "run-b"
    assert recommendation.changes == {"learning_rate": 2.5e-4}
    assert recommendation.keep_constant == ("batch_size",)
    assert recommendation.success_criteria[0].operator == ">="


def test_suggestion_accepts_an_explicit_objective():
    recommendation = suggest_next_experiment(
        [ExperimentRun(name="run", metrics={"recall": 0.7}, parameters={})],
        objective_metric="recall",
    )

    assert recommendation.success_criteria[0].metric == "recall"


@pytest.mark.parametrize(
    ("runs", "kwargs", "message"),
    [
        ([], {}, "at least one"),
        ([ExperimentRun("run", {"loss": 1.0})], {"minimum_improvement": 0}, "positive"),
        ([ExperimentRun("run", {"custom": 1.0})], {}, "no supported objective"),
    ],
)
def test_suggestion_rejects_insufficient_evidence(runs, kwargs, message):
    with pytest.raises(ValueError, match=message):
        suggest_next_experiment(runs, **kwargs)
