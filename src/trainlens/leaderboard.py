"""Multi-run leaderboards for portable TrainLens experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite

from trainlens.experiments import ExperimentRun, MetricConstraint, ObjectiveSpec, pareto_front
from trainlens.metric_semantics import MetricDirection, metric_direction
from trainlens.models.run import TrainingRun
from trainlens.run_metrics import final_metric_values


@dataclass(frozen=True)
class LeaderboardRow:
    """One run in a multi-run leaderboard."""

    rank: int | None
    run_id: str
    model_name: str | None
    metrics: Mapping[str, float | None]
    pareto: bool
    satisfies_constraints: bool
    missing_objectives: tuple[str, ...] = ()


@dataclass(frozen=True)
class RunLeaderboard:
    """Ranked view over several portable training runs."""

    rows: tuple[LeaderboardRow, ...]
    objectives: tuple[ObjectiveSpec, ...]
    constraints: tuple[MetricConstraint, ...] = ()

    def to_markdown(self) -> str:
        """Render the leaderboard as a compact Markdown table."""

        metrics = tuple(objective.metric for objective in self.objectives)
        header = ["Rank", "Run", *metrics, "Pareto", "Eligible", "Missing objectives"]
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join("---" for _ in header) + " |",
        ]
        for row in self.rows:
            values = [
                "" if row.rank is None else str(row.rank),
                row.run_id,
                *(_format_metric(row.metrics.get(metric)) for metric in metrics),
                "yes" if row.pareto else "",
                "yes" if row.satisfies_constraints else "no",
                ", ".join(row.missing_objectives),
            ]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines) + "\n"

    def _repr_markdown_(self) -> str:
        return self.to_markdown()


def leaderboard(
    runs: Sequence[TrainingRun],
    objectives: Sequence[ObjectiveSpec],
    *,
    constraints: Sequence[MetricConstraint] = (),
) -> RunLeaderboard:
    """Rank runs by the primary objective and mark the Pareto-efficient subset."""

    if not objectives:
        raise ValueError("at least one objective is required")
    _ensure_unique_ids(runs)
    objective_tuple = tuple(objectives)
    constraint_tuple = tuple(constraints)
    direction = _objective_direction(objective_tuple[0])
    experiments = tuple(_as_experiment_run(run) for run in runs)
    frontier_ids = {
        run.name for run in pareto_front(experiments, objective_tuple, constraints=constraint_tuple)
    }
    primary = objective_tuple[0].metric
    prepared = []
    for run in runs:
        metrics = _final_metrics(run)
        missing = tuple(
            objective.metric for objective in objective_tuple if objective.metric not in metrics
        )
        eligible = not missing and _satisfies_constraints(metrics, constraint_tuple)
        prepared.append((run, metrics, eligible, missing))
    prepared.sort(key=lambda item: _sort_key(item[1].get(primary), item[2], direction))

    rank = 0
    rows: list[LeaderboardRow] = []
    for run, metrics, eligible, missing in prepared:
        current_rank: int | None = None
        if eligible and metrics.get(primary) is not None:
            rank += 1
            current_rank = rank
        rows.append(
            LeaderboardRow(
                rank=current_rank,
                run_id=run.run_id,
                model_name=run.model_name,
                metrics={
                    objective.metric: metrics.get(objective.metric)
                    for objective in objective_tuple
                },
                pareto=run.run_id in frontier_ids,
                satisfies_constraints=eligible,
                missing_objectives=missing,
            )
        )
    return RunLeaderboard(tuple(rows), objective_tuple, constraint_tuple)


def _final_metrics(run: TrainingRun) -> dict[str, float]:
    return final_metric_values(run)


def _as_experiment_run(run: TrainingRun) -> ExperimentRun:
    return ExperimentRun(name=run.run_id, metrics=_final_metrics(run), parameters=run.parameters)


def _objective_direction(objective: ObjectiveSpec) -> MetricDirection:
    if objective.direction == "min":
        return "lower"
    if objective.direction == "max":
        return "higher"
    direction = metric_direction(objective.metric)
    if direction is None:
        raise ValueError(f"cannot infer whether {objective.metric!r} should increase or decrease")
    return direction


def _satisfies_constraints(
    metrics: Mapping[str, float], constraints: Sequence[MetricConstraint]
) -> bool:
    for constraint in constraints:
        value = metrics.get(constraint.metric)
        if value is None:
            return False
        if constraint.operator == "<=" and value > constraint.threshold:
            return False
        if constraint.operator == ">=" and value < constraint.threshold:
            return False
    return True


def _sort_key(
    value: float | None, eligible: bool, direction: MetricDirection
) -> tuple[int, float]:
    if not eligible or value is None:
        return 1, 0.0
    return 0, value if direction == "lower" else -value


def _ensure_unique_ids(runs: Sequence[TrainingRun]) -> None:
    ids = [run.run_id for run in runs]
    if len(ids) != len(set(ids)):
        raise ValueError("leaderboard requires unique run_id values")


def _format_metric(value: float | None) -> str:
    return "" if value is None else f"{value:.6g}"
