"""Small command-line interface for comparing and gating portable runs."""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from pathlib import Path

from trainlens.comparison import compare_runs, render_run_comparison
from trainlens.models.run import TrainingRun
from trainlens.runs import load_run

_GATE_PATTERN = re.compile(
    r"^\s*(?P<metric>[^<>=]+?)\s*(?P<operator><=|>=|==|<|>)\s*"
    r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the TrainLens CLI and return a process exit code."""

    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "compare":
        return _compare(args.baseline, args.candidate)
    if args.command == "check":
        return _check(args.candidate, args.against, tuple(args.requirement))
    parser.error("a command is required")
    return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trainlens")
    subparsers = parser.add_subparsers(dest="command")

    compare = subparsers.add_parser("compare", help="compare two portable TrainingRun JSON files")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)

    check = subparsers.add_parser("check", help="fail on material regressions or metric gates")
    check.add_argument("candidate", type=Path)
    check.add_argument("--against", type=Path, default=None)
    check.add_argument(
        "--require",
        dest="requirement",
        action="append",
        default=[],
        help="metric gate such as 'f1>=0.85' or 'latency_ms<=10'",
    )
    return parser


def _compare(baseline_path: Path, candidate_path: Path) -> int:
    baseline = load_run(baseline_path)
    candidate = load_run(candidate_path)
    comparison = compare_runs(
        baseline,
        candidate,
        baseline_name=baseline_path.stem,
        experiment_name=candidate_path.stem,
    )
    print(render_run_comparison(comparison), end="")
    return 0


def _check(
    candidate_path: Path,
    baseline_path: Path | None,
    requirements: tuple[str, ...],
) -> int:
    candidate = load_run(candidate_path)
    failures: list[str] = []
    if baseline_path is not None:
        baseline = load_run(baseline_path)
        comparison = compare_runs(
            baseline,
            candidate,
            baseline_name=baseline_path.stem,
            experiment_name=candidate_path.stem,
        )
        material = [item for item in comparison.regressions if item.magnitude == "material"]
        failures.extend(
            f"material regression: {item.name} "
            f"{_format_value(item.baseline)} -> {_format_value(item.experiment)}"
            for item in material
        )

    metrics = _final_metrics(candidate)
    for expression in requirements:
        metric, operator, threshold = _parse_gate(expression)
        value = metrics.get(metric)
        if value is None:
            failures.append(f"missing required metric: {metric}")
            continue
        if not _passes(value, operator, threshold):
            failures.append(
                f"gate failed: {metric}={value:.6g} does not satisfy {operator}{threshold:.6g}"
            )

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("PASS: all TrainLens checks satisfied")
    return 0


def _parse_gate(expression: str) -> tuple[str, str, float]:
    match = _GATE_PATTERN.match(expression)
    if match is None:
        raise ValueError(
            f"invalid metric gate {expression!r}; expected syntax such as 'f1>=0.85'"
        )
    return match.group("metric").strip(), match.group("operator"), float(match.group("value"))


def _passes(value: float, operator: str, threshold: float) -> bool:
    if operator == "<=":
        return value <= threshold
    if operator == ">=":
        return value >= threshold
    if operator == "<":
        return value < threshold
    if operator == ">":
        return value > threshold
    return value == threshold


def _final_metrics(run: TrainingRun) -> dict[str, float]:
    output: dict[str, float] = {}
    for series in run.metrics:
        value = series.last
        if value is not None:
            output[series.name] = value
    return output


def _format_value(value: float | None) -> str:
    return "missing" if value is None else f"{value:.6g}"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
