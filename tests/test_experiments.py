import math

import pytest

from trainlens import (
    ExperimentRun,
    experiment_config,
    render_next_experiment,
    suggest_next_experiment,
)


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


def test_suggestion_avoids_a_noop_when_dropout_is_already_at_limit():
    recommendation = suggest_next_experiment(
        [
            ExperimentRun(
                name="regularized",
                metrics={"train_loss": 0.2, "validation_loss": 0.5},
                parameters={"dropout": 0.8},
            )
        ]
    )

    assert recommendation.changes == {"weight_decay": 0.01}


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


def test_suggestion_ignores_non_finite_objective_values():
    recommendation = suggest_next_experiment(
        [
            ExperimentRun(name="invalid", metrics={"loss": math.nan}),
            ExperimentRun(name="valid", metrics={"loss": 0.4}),
        ]
    )

    assert recommendation.source_run == "valid"
    assert math.isfinite(recommendation.success_criteria[0].target)


def test_suggestion_falls_back_when_higher_priority_objective_is_non_finite():
    recommendation = suggest_next_experiment(
        [
            ExperimentRun(
                name="run",
                metrics={"validation_loss": math.nan, "accuracy": 0.8},
                parameters={"learning_rate": 1e-3},
            )
        ]
    )

    assert recommendation.success_criteria[0].metric == "accuracy"
    assert recommendation.success_criteria[0].operator == ">="


def test_recommendation_can_apply_its_single_change_to_a_base_config():
    base = {"learning_rate": 1e-3, "batch_size": 16, "epochs": 5}
    recommendation = suggest_next_experiment(
        [ExperimentRun(name="run", metrics={"accuracy": 0.8}, parameters=base)]
    )

    config = experiment_config(recommendation, base_parameters=base)

    assert config == {"learning_rate": 5e-4, "batch_size": 16, "epochs": 5}
    assert base["learning_rate"] == 1e-3


def test_recommendation_renders_as_reviewable_markdown():
    recommendation = suggest_next_experiment(
        [
            ExperimentRun(
                name="baseline",
                metrics={"train_loss": 0.2, "val_loss": 0.5},
                parameters={"dropout": 0.1, "batch_size": 32},
            )
        ]
    )

    markdown = render_next_experiment(recommendation)

    assert "## TrainLens Next Experiment" in markdown
    assert "**Source run:** baseline" in markdown
    assert "`dropout`: `0.15`" in markdown
    assert "`val_loss` <= `0.495`" in markdown


@pytest.mark.parametrize(
    ("runs", "kwargs", "message"),
    [
        ([], {}, "at least one"),
        ([ExperimentRun("run", {"loss": 1.0})], {"minimum_improvement": 0}, "positive"),
        (
            [ExperimentRun("run", {"loss": 1.0})],
            {"minimum_improvement": math.nan},
            "finite and positive",
        ),
        ([ExperimentRun("run", {"loss": math.inf})], {}, "non-finite"),
        ([ExperimentRun("run", {"custom": 1.0})], {}, "no supported objective"),
    ],
)
def test_suggestion_rejects_insufficient_evidence(runs, kwargs, message):
    with pytest.raises(ValueError, match=message):
        suggest_next_experiment(runs, **kwargs)
