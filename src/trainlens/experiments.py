"""Evidence-backed recommendations for the next controlled experiment."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from typing import Literal, TypeAlias

ParameterValue: TypeAlias = str | int | float | bool | None
EstimatedCost = Literal["low", "medium", "high", "unknown"]

_LOWER_IS_BETTER = ("loss", "error", "perplexity", "wer", "cer", "latency")
_HIGHER_IS_BETTER = ("accuracy", "acc", "auc", "f1", "precision", "recall", "score")
_OBJECTIVE_PRIORITY = (
    "validation_loss",
    "val_loss",
    "eval_loss",
    "validation_accuracy",
    "val_accuracy",
    "eval_accuracy",
    "f1",
    "accuracy",
    "loss",
)


@dataclass(frozen=True)
class ExperimentRun:
    """Metrics and configuration needed to reason about one experiment."""

    name: str
    metrics: Mapping[str, float]
    parameters: Mapping[str, ParameterValue] = field(default_factory=dict)
    estimated_cost: EstimatedCost = "unknown"


@dataclass(frozen=True)
class SuccessCriterion:
    """A measurable condition for accepting a proposed experiment."""

    metric: str
    operator: Literal["<=", ">="]
    target: float


@dataclass(frozen=True)
class NextExperimentRecommendation:
    """A controlled, evidence-linked proposal for the next training run."""

    hypothesis: str
    changes: Mapping[str, ParameterValue]
    keep_constant: tuple[str, ...]
    success_criteria: tuple[SuccessCriterion, ...]
    estimated_cost: EstimatedCost
    confidence: float
    evidence: tuple[str, ...]
    source_run: str


def suggest_next_experiment(
    runs: Sequence[ExperimentRun],
    *,
    objective_metric: str | None = None,
    minimum_improvement: float = 0.01,
) -> NextExperimentRecommendation:
    """Suggest one controlled follow-up from one or more completed runs."""

    if not runs:
        raise ValueError("at least one experiment run is required")
    if not isfinite(minimum_improvement) or minimum_improvement <= 0:
        raise ValueError("minimum_improvement must be finite and positive")
    objective = objective_metric or _select_objective(runs)
    if objective is None:
        if any(name in run.metrics for run in runs for name in _OBJECTIVE_PRIORITY):
            raise ValueError("supported objective metrics are non-finite or non-numeric")
        raise ValueError("no supported objective metric was found; pass objective_metric")
    direction = _metric_direction(objective)
    if direction is None:
        raise ValueError(f"cannot infer whether {objective!r} should increase or decrease")
    eligible = [
        run
        for run in runs
        if objective in run.metrics and _is_finite_metric(run.metrics[objective])
    ]
    if not eligible:
        raise ValueError(
            f"objective metric {objective!r} is missing or non-finite in every run"
        )
    best = min(eligible, key=lambda run: run.metrics[objective]) if direction == "lower" else max(
        eligible, key=lambda run: run.metrics[objective]
    )
    change, hypothesis, evidence, confidence = _propose_change(best, objective)
    keep_constant = tuple(sorted(name for name in best.parameters if name not in change))
    target = _target(best.metrics[objective], direction, minimum_improvement)
    return NextExperimentRecommendation(
        hypothesis=hypothesis,
        changes=change,
        keep_constant=keep_constant,
        success_criteria=(
            SuccessCriterion(
                metric=objective,
                operator="<=" if direction == "lower" else ">=",
                target=target,
            ),
        ),
        estimated_cost=best.estimated_cost,
        confidence=confidence,
        evidence=evidence,
        source_run=best.name,
    )


def experiment_config(
    recommendation: NextExperimentRecommendation,
    *,
    base_parameters: Mapping[str, ParameterValue] | None = None,
) -> dict[str, ParameterValue]:
    """Build an executable parameter mapping with recommendation changes applied."""

    config = dict(base_parameters or {})
    config.update(recommendation.changes)
    return config


def render_next_experiment(recommendation: NextExperimentRecommendation) -> str:
    """Render a recommendation as reviewable Markdown."""

    lines = [
        "## TrainLens Next Experiment",
        "",
        f"**Source run:** {recommendation.source_run}",
        f"**Estimated cost:** {recommendation.estimated_cost}",
        f"**Confidence:** {recommendation.confidence:.0%}",
        "",
        "### Hypothesis",
        recommendation.hypothesis,
        "",
        "### Change one variable",
    ]
    lines.extend(f"- `{name}`: `{value}`" for name, value in recommendation.changes.items())
    lines.extend(["", "### Keep constant"])
    lines.extend(f"- `{name}`" for name in recommendation.keep_constant)
    lines.extend(["", "### Success criteria"])
    lines.extend(
        f"- `{criterion.metric}` {criterion.operator} `{criterion.target:.6g}`"
        for criterion in recommendation.success_criteria
    )
    lines.extend(["", "### Evidence"])
    lines.extend(f"- {item}" for item in recommendation.evidence)
    return "\n".join(lines).strip() + "\n"


def _select_objective(runs: Sequence[ExperimentRun]) -> str | None:
    available = {
        name
        for run in runs
        for name, value in run.metrics.items()
        if _is_finite_metric(value)
    }
    return next((name for name in _OBJECTIVE_PRIORITY if name in available), None)


def _is_finite_metric(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and isfinite(value)
    )


def _metric_direction(name: str) -> Literal["lower", "higher"] | None:
    tokens = set(re.findall(r"[a-z0-9]+", name.lower()))
    if tokens.intersection(_LOWER_IS_BETTER):
        return "lower"
    if tokens.intersection(_HIGHER_IS_BETTER):
        return "higher"
    return None


def _propose_change(
    run: ExperimentRun, objective: str
) -> tuple[dict[str, ParameterValue], str, tuple[str, ...], float]:
    train_loss = _first_value(run.metrics, ("train_loss", "training_loss", "loss"))
    validation_loss = _first_value(
        run.metrics, ("validation_loss", "val_loss", "eval_loss")
    )
    if train_loss is not None and validation_loss is not None and validation_loss > train_loss:
        dropout = run.parameters.get("dropout")
        if (
            isinstance(dropout, int | float)
            and not isinstance(dropout, bool)
            and 0 <= dropout < 0.8
        ):
            old = float(dropout)
            new = min(0.8, round(old + 0.05, 4))
            return (
                {"dropout": new},
                "A small increase in dropout may reduce the observed generalization gap.",
                (f"training loss={train_loss}", f"validation loss={validation_loss}"),
                0.72,
            )
        return (
            {"weight_decay": 0.01},
            "Adding a controlled amount of weight decay may reduce the generalization gap.",
            (f"training loss={train_loss}", f"validation loss={validation_loss}"),
            0.62,
        )
    learning_rate = run.parameters.get("learning_rate")
    if isinstance(learning_rate, int | float) and not isinstance(learning_rate, bool):
        new_rate = float(learning_rate) * 0.5
        return (
            {"learning_rate": new_rate},
            "A lower learning rate is a low-dimensional test of whether optimization can improve.",
            (f"best observed {objective}={run.metrics[objective]}",),
            0.5,
        )
    return (
        {"learning_rate_multiplier": 0.5},
        "Test a lower learning rate while holding the recorded configuration constant.",
        (f"best observed {objective}={run.metrics[objective]}",),
        0.4,
    )


def _first_value(metrics: Mapping[str, float], names: tuple[str, ...]) -> float | None:
    for name in names:
        if name in metrics:
            return float(metrics[name])
    return None


def _target(value: float, direction: Literal["lower", "higher"], improvement: float) -> float:
    if direction == "lower":
        return value - abs(value) * improvement
    return value + abs(value) * improvement
