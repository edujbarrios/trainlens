"""Evidence-backed recommendations for the next controlled experiment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from typing import Literal, TypeAlias

from trainlens.metric_semantics import metric_bounds, metric_direction

ParameterValue: TypeAlias = str | int | float | bool | None
EstimatedCost = Literal["low", "medium", "high", "unknown"]
ObjectiveDirection = Literal["min", "max"]

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
class ObjectiveSpec:
    """One metric to optimize in a multiobjective experiment search."""

    metric: str
    direction: ObjectiveDirection | None = None


@dataclass(frozen=True)
class MetricConstraint:
    """Hard metric requirement applied before Pareto ranking."""

    metric: str
    operator: Literal["<=", ">="]
    threshold: float


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
    _validate_improvement(minimum_improvement)
    objective = objective_metric or _select_objective(runs)
    if objective is None:
        if any(name in run.metrics for run in runs for name in _OBJECTIVE_PRIORITY):
            raise ValueError("supported objective metrics are non-finite or non-numeric")
        raise ValueError("no supported objective metric was found; pass objective_metric")
    direction = metric_direction(objective)
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
    best = (
        min(eligible, key=lambda run: run.metrics[objective])
        if direction == "lower"
        else max(eligible, key=lambda run: run.metrics[objective])
    )
    change, hypothesis, evidence, confidence = _propose_change(best, objective)
    keep_constant = tuple(sorted(name for name in best.parameters if name not in change))
    target = _target(best.metrics[objective], direction, minimum_improvement, objective)
    if target == best.metrics[objective]:
        raise ValueError(f"objective metric {objective!r} is already at its natural optimum")
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


def pareto_front(
    runs: Sequence[ExperimentRun],
    objectives: Sequence[ObjectiveSpec],
    *,
    constraints: Sequence[MetricConstraint] = (),
) -> tuple[ExperimentRun, ...]:
    """Return non-dominated runs that satisfy all hard constraints."""

    if not runs:
        return ()
    if not objectives:
        raise ValueError("at least one objective is required")
    directions = tuple(_objective_direction(objective) for objective in objectives)
    eligible = [
        run
        for run in runs
        if _has_objective_metrics(run, objectives) and _satisfies_constraints(run, constraints)
    ]
    frontier: list[ExperimentRun] = []
    for index, candidate in enumerate(eligible):
        dominated = any(
            other_index != index
            and _dominates(other, candidate, objectives, directions)
            for other_index, other in enumerate(eligible)
        )
        if not dominated:
            frontier.append(candidate)
    return tuple(frontier)


def suggest_multiobjective_experiment(
    runs: Sequence[ExperimentRun],
    objectives: Sequence[ObjectiveSpec],
    *,
    constraints: Sequence[MetricConstraint] = (),
    minimum_improvement: float = 0.01,
) -> NextExperimentRecommendation:
    """Suggest a controlled follow-up from the balanced Pareto-efficient run."""

    _validate_improvement(minimum_improvement)
    frontier = pareto_front(runs, objectives, constraints=constraints)
    if not frontier:
        raise ValueError("no run satisfies the objectives and constraints")
    source = _balanced_frontier_run(frontier, objectives)
    primary = objectives[0]
    primary_direction = _objective_direction(primary)
    change, hypothesis, evidence, confidence = _propose_change(source, primary.metric)
    criteria = tuple(
        _criterion_for_objective(source, objective, minimum_improvement)
        for objective in objectives
    )
    keep_constant = tuple(sorted(name for name in source.parameters if name not in change))
    objective_names = ", ".join(objective.metric for objective in objectives)
    constraint_evidence = tuple(
        f"constraint {item.metric} {item.operator} {item.threshold:g}" for item in constraints
    )
    direction_text = "lower" if primary_direction == "lower" else "higher"
    return NextExperimentRecommendation(
        hypothesis=(
            hypothesis
            + f" The source run is Pareto-efficient; keep the trade-off while pushing "
            f"{primary.metric} {direction_text}."
        ),
        changes=change,
        keep_constant=keep_constant,
        success_criteria=criteria,
        estimated_cost=source.estimated_cost,
        confidence=min(confidence, 0.65),
        evidence=(
            *evidence,
            f"Pareto-efficient across: {objective_names}",
            *constraint_evidence,
        ),
        source_run=source.name,
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


def _validate_improvement(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("minimum_improvement must be a number")
    if not isfinite(value) or value <= 0:
        raise ValueError("minimum_improvement must be finite and positive")


def _objective_direction(objective: ObjectiveSpec) -> Literal["lower", "higher"]:
    if objective.direction == "min":
        return "lower"
    if objective.direction == "max":
        return "higher"
    inferred = metric_direction(objective.metric)
    if inferred is None:
        raise ValueError(
            f"cannot infer whether {objective.metric!r} should increase or decrease; "
            "set ObjectiveSpec.direction"
        )
    return inferred


def _has_objective_metrics(run: ExperimentRun, objectives: Sequence[ObjectiveSpec]) -> bool:
    return all(
        objective.metric in run.metrics and _is_finite_metric(run.metrics[objective.metric])
        for objective in objectives
    )


def _satisfies_constraints(
    run: ExperimentRun,
    constraints: Sequence[MetricConstraint],
) -> bool:
    for constraint in constraints:
        value = run.metrics.get(constraint.metric)
        if value is None or not _is_finite_metric(value):
            return False
        if constraint.operator == "<=" and value > constraint.threshold:
            return False
        if constraint.operator == ">=" and value < constraint.threshold:
            return False
    return True


def _dominates(
    left: ExperimentRun,
    right: ExperimentRun,
    objectives: Sequence[ObjectiveSpec],
    directions: Sequence[Literal["lower", "higher"]],
) -> bool:
    no_worse = True
    strictly_better = False
    for objective, direction in zip(objectives, directions, strict=True):
        left_value = left.metrics[objective.metric]
        right_value = right.metrics[objective.metric]
        if direction == "lower":
            no_worse = no_worse and left_value <= right_value
            strictly_better = strictly_better or left_value < right_value
        else:
            no_worse = no_worse and left_value >= right_value
            strictly_better = strictly_better or left_value > right_value
    return no_worse and strictly_better


def _balanced_frontier_run(
    frontier: Sequence[ExperimentRun],
    objectives: Sequence[ObjectiveSpec],
) -> ExperimentRun:
    scores = [0] * len(frontier)
    for objective in objectives:
        direction = _objective_direction(objective)
        order = sorted(
            range(len(frontier)),
            key=lambda index: frontier[index].metrics[objective.metric],
            reverse=direction == "higher",
        )
        for rank, index in enumerate(order):
            scores[index] += rank
    best_index = min(range(len(frontier)), key=lambda index: scores[index])
    return frontier[best_index]


def _criterion_for_objective(
    run: ExperimentRun,
    objective: ObjectiveSpec,
    improvement: float,
) -> SuccessCriterion:
    direction = _objective_direction(objective)
    return SuccessCriterion(
        metric=objective.metric,
        operator="<=" if direction == "lower" else ">=",
        target=_target(run.metrics[objective.metric], direction, improvement, objective.metric),
    )


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
        weight_decay = run.parameters.get("weight_decay")
        if weight_decay is None:
            return (
                {"weight_decay": 0.01},
                "Adding a controlled amount of weight decay may reduce the generalization gap.",
                (f"training loss={train_loss}", f"validation loss={validation_loss}"),
                0.62,
            )
        if (
            isinstance(weight_decay, int | float)
            and not isinstance(weight_decay, bool)
            and isfinite(weight_decay)
            and 0 <= weight_decay < 0.2
        ):
            old_weight_decay = float(weight_decay)
            new_weight_decay = (
                0.01
                if old_weight_decay < 0.01
                else min(0.2, round(old_weight_decay * 2.0, 6))
            )
            if new_weight_decay > old_weight_decay:
                return (
                    {"weight_decay": new_weight_decay},
                    "A controlled increase in weight decay may reduce the generalization gap.",
                    (
                        f"training loss={train_loss}",
                        f"validation loss={validation_loss}",
                        f"current weight_decay={old_weight_decay:g}",
                    ),
                    0.62,
                )
    learning_rate = run.parameters.get("learning_rate")
    if (
        isinstance(learning_rate, int | float)
        and not isinstance(learning_rate, bool)
        and isfinite(learning_rate)
        and learning_rate > 0
    ):
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


def _target(
    value: float,
    direction: Literal["lower", "higher"],
    improvement: float,
    metric: str,
) -> float:
    change = max(abs(value) * improvement, improvement if value == 0 else 0.0)
    lower_bound, upper_bound = metric_bounds(metric)
    if upper_bound == 1.0 and value > 1.0:
        upper_bound = 100.0 if value <= 100.0 else None
    target = value - change if direction == "lower" else value + change
    if lower_bound is not None:
        target = max(lower_bound, target)
    if upper_bound is not None:
        target = min(upper_bound, target)
    return target
