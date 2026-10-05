"""Small command-line interface for portable runs, CI gates, and agent workflows."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from pathlib import Path

from trainlens.agent import AgentContext, build_agent_context_from_run, verify_agent_plan
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
    if args.command == "agent-context":
        return _agent_context(args.run, args.format, args.output, args.objective)
    if args.command == "verify-agent-plan":
        return _verify_agent_plan(args.plan, args.context)
    parser.error("a command is required")


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

    agent_context = subparsers.add_parser(
        "agent-context",
        help="build provider-free evidence context for an external coding or research agent",
    )
    agent_context.add_argument("run", type=Path, help="portable TrainingRun JSON file")
    agent_context.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="json",
        help="output format; JSON is recommended for skills and agents",
    )
    agent_context.add_argument("--output", type=Path, default=None)
    agent_context.add_argument("--objective", default="propose_next_experiment")

    verify_plan = subparsers.add_parser(
        "verify-agent-plan",
        help="verify an external agent plan against TrainLens evidence IDs",
    )
    verify_plan.add_argument("plan", type=Path, help="agent response JSON")
    verify_plan.add_argument(
        "--context",
        type=Path,
        required=True,
        help="JSON produced by 'trainlens agent-context'",
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


def _agent_context(
    run_path: Path,
    output_format: str,
    output_path: Path | None,
    objective: str,
) -> int:
    context = build_agent_context_from_run(load_run(run_path), objective=objective)
    rendered = context.to_json() if output_format == "json" else context.to_markdown()
    _write_or_print(rendered, output_path)
    return 0


def _verify_agent_plan(plan_path: Path, context_path: Path) -> int:
    context_payload = json.loads(context_path.read_text(encoding="utf-8"))
    if not isinstance(context_payload, dict):
        raise ValueError("agent context JSON root must be an object")
    context = AgentContext.from_dict(context_payload)
    response = plan_path.read_text(encoding="utf-8")
    plan = verify_agent_plan(response, context=context)
    payload = {
        "fully_supported": plan.is_fully_supported,
        "unsupported_evidence_ids": list(plan.unsupported_evidence_ids),
        "recommendations": [
            {
                "action": recommendation.action,
                "evidence_ids": list(recommendation.evidence_ids),
                "supported": recommendation.is_supported,
            }
            for recommendation in plan.recommendations
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if plan.is_fully_supported else 1


def _write_or_print(content: str, output_path: Path | None) -> None:
    rendered = content.rstrip() + "\n"
    if output_path is None:
        print(rendered, end="")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")


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
