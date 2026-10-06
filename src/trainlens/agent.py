"""Provider-free context and verification helpers for external ML agents."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

from IPython import get_ipython

from trainlens.analysis_config import AnalysisConfig
from trainlens.dataset import DatasetExplanation
from trainlens.llm.context import ContextPolicy, build_llm_notebook_context_from_snapshot
from trainlens.llm.dataset_context import append_dataset_explanation
from trainlens.llm.verified import (
    LLMEvidenceItem,
    VerifiedImprovementPlan,
    evidence_catalog,
    parse_verified_improvement_plan,
)
from trainlens.models.run import TrainingRun
from trainlens.pipeline import analyze_snapshot, snapshot_namespace
from trainlens.run_metrics import metric_namespace_from_run

_AGENT_INSTRUCTIONS = (
    "Use only the supplied deterministic TrainLens evidence when claiming observed facts.",
    "Distinguish observed evidence from hypotheses and expected effects.",
    "Do not tune against held-out test evidence; use test results only as final "
    "generalization evidence.",
    "Prefer one controlled, low-cost, high-information change before broad search.",
    "Cite TrainLens evidence IDs for every recommendation.",
    "Define a measurable success criterion before proposing execution.",
)

_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "required": ["recommendations"],
    "properties": {
        "recommendations": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": [
                    "action",
                    "rationale",
                    "evidence_ids",
                    "confidence",
                    "success_criterion",
                ],
                "properties": {
                    "action": {"type": "string"},
                    "rationale": {"type": "string"},
                    "evidence_ids": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string"},
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "success_criterion": {"type": "string"},
                },
            },
        }
    },
}


@dataclass(frozen=True)
class AgentContext:
    """Deterministic TrainLens evidence prepared for an already-running agent."""

    objective: str
    markdown: str
    metrics: Mapping[str, float]
    evidence: tuple[LLMEvidenceItem, ...]
    instructions: tuple[str, ...]
    output_schema: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(self, "output_schema", _freeze_mapping(self.output_schema))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation for tools and skills."""

        return {
            "format_version": 1,
            "mode": "agent",
            "objective": self.objective,
            "metrics": dict(self.metrics),
            "evidence": [
                {"evidence_id": item.evidence_id, "detail": item.detail}
                for item in self.evidence
            ],
            "instructions": list(self.instructions),
            "output_schema": _thaw_mapping(self.output_schema),
            "markdown": self.markdown,
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Render provider-free agent context as JSON."""

        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def to_markdown(self) -> str:
        """Render the same bounded context as Markdown."""

        return self.markdown

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> AgentContext:
        """Restore an :class:`AgentContext` previously serialized with :meth:`to_dict`."""

        objective = _required_text(payload.get("objective"), "objective")
        markdown = _required_text(payload.get("markdown"), "markdown")
        metrics = _metrics_from_payload(payload.get("metrics"))
        evidence = _evidence_from_payload(payload.get("evidence"))
        instructions = _strings_from_payload(payload.get("instructions"), "instructions")
        output_schema = payload.get("output_schema")
        if not isinstance(output_schema, dict):
            raise ValueError("agent context output_schema must be an object")
        return cls(
            objective=objective,
            markdown=markdown,
            metrics=metrics,
            evidence=evidence,
            instructions=instructions,
            output_schema=dict(output_schema),
        )


def build_agent_context(
    namespace: Mapping[str, Any] | None = None,
    *,
    analysis_config: AnalysisConfig | None = None,
    context_policy: ContextPolicy | None = None,
    max_metric_points: int = 12,
    include_values: bool = False,
    dataset_explanation: DatasetExplanation | None = None,
    objective: str = "propose_next_experiment",
) -> AgentContext:
    """Build bounded deterministic context without making an LLM or network request."""

    report_namespace = _current_user_namespace() if namespace is None else namespace
    return _build_agent_context(
        report_namespace,
        analysis_config=analysis_config,
        context_policy=context_policy,
        max_metric_points=max_metric_points,
        include_values=include_values,
        dataset_explanation=dataset_explanation,
        objective=objective,
    )


def build_agent_context_from_run(
    run: TrainingRun,
    *,
    context_policy: ContextPolicy | None = None,
    max_metric_points: int = 12,
    objective: str = "propose_next_experiment",
) -> AgentContext:
    """Build agent context from one portable :class:`TrainingRun` for CLI/skill workflows."""

    namespace = metric_namespace_from_run(run)
    extra_evidence = _portable_run_evidence(run)
    return _build_agent_context(
        namespace,
        analysis_config=None,
        context_policy=context_policy,
        max_metric_points=max_metric_points,
        include_values=False,
        dataset_explanation=None,
        objective=objective,
        extra_evidence=extra_evidence,
    )


def verify_agent_plan(
    response: str | Mapping[str, object],
    *,
    context: AgentContext | None = None,
    evidence: Sequence[LLMEvidenceItem] | None = None,
) -> VerifiedImprovementPlan:
    """Verify structured agent recommendations against deterministic TrainLens evidence IDs."""

    if context is not None and evidence is not None:
        raise ValueError("pass either context or evidence, not both")
    catalog = context.evidence if context is not None else tuple(evidence or ())
    if not catalog:
        raise ValueError("agent plan verification requires TrainLens evidence")
    raw_response = (
        json.dumps(dict(response), sort_keys=True)
        if isinstance(response, Mapping)
        else response
    )
    return parse_verified_improvement_plan(raw_response, evidence=tuple(catalog))


def _build_agent_context(
    namespace: Mapping[str, Any],
    *,
    analysis_config: AnalysisConfig | None,
    context_policy: ContextPolicy | None,
    max_metric_points: int,
    include_values: bool,
    dataset_explanation: DatasetExplanation | None,
    objective: str,
    extra_evidence: tuple[LLMEvidenceItem, ...] = (),
) -> AgentContext:
    snapshot = snapshot_namespace(namespace)
    result = analyze_snapshot(snapshot, config=analysis_config)
    notebook_context = build_llm_notebook_context_from_snapshot(
        snapshot,
        max_metric_points=max_metric_points,
        include_values=include_values,
        analysis_config=analysis_config,
        deterministic_result=result,
        context_policy=context_policy,
    )
    notebook_context = append_dataset_explanation(
        notebook_context,
        dataset_explanation=dataset_explanation,
        context_policy=context_policy,
    )
    evidence = evidence_catalog(result, dataset_explanation=dataset_explanation) + extra_evidence
    instructions = _AGENT_INSTRUCTIONS
    markdown = _compose_markdown(
        notebook_context.markdown,
        objective=objective,
        evidence=evidence,
        instructions=instructions,
        output_schema=_OUTPUT_SCHEMA,
    )
    return AgentContext(
        objective=objective,
        markdown=markdown,
        metrics=dict(notebook_context.metrics),
        evidence=evidence,
        instructions=instructions,
        output_schema=_OUTPUT_SCHEMA,
    )


def _compose_markdown(
    notebook_markdown: str,
    *,
    objective: str,
    evidence: tuple[LLMEvidenceItem, ...],
    instructions: tuple[str, ...],
    output_schema: Mapping[str, object],
) -> str:
    lines = [
        "# TrainLens Agent Context",
        "",
        f"Objective: `{objective}`",
        "",
        "TrainLens made no LLM request to produce this context. "
        "The surrounding agent is the reasoning runtime.",
        "",
        notebook_markdown.strip(),
        "",
        "## Evidence IDs",
        "",
    ]
    if evidence:
        lines.extend(f"- `{item.evidence_id}`: {item.detail}" for item in evidence)
    else:
        lines.append("- No deterministic evidence IDs are available.")
    lines.extend(["", "## Agent Instructions", ""])
    lines.extend(f"- {instruction}" for instruction in instructions)
    lines.extend(
        [
            "",
            "## Expected Output Schema",
            "",
            "Return JSON matching this schema so TrainLens can verify evidence references:",
            "",
            "```json",
            json.dumps(_thaw_mapping(output_schema), indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _portable_run_evidence(run: TrainingRun) -> tuple[LLMEvidenceItem, ...]:
    items = [LLMEvidenceItem("run:id", f"run_id={run.run_id}")]
    if run.model_name:
        items.append(LLMEvidenceItem("run:model", f"model_name={run.model_name}"))
    if run.framework:
        items.append(LLMEvidenceItem("run:framework", f"framework={run.framework}"))
    for name, value in sorted(run.parameters.items()):
        items.append(
            LLMEvidenceItem(
                f"run:parameter:{_evidence_slug(name)}",
                f"{name}={value!r}",
            )
        )
    for index, note in enumerate(run.notes, start=1):
        items.append(LLMEvidenceItem(f"run:note:{index}", note))
    return tuple(items)


def _evidence_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return slug or "item"


def _freeze_value(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(_freeze_value(item) for item in value)
    return value


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(
        {str(key): _freeze_value(item) for key, item in value.items()}
    )


def _thaw_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_value(item) for item in value]
    return value


def _thaw_mapping(value: Mapping[str, object]) -> dict[str, object]:
    return {str(key): _thaw_value(item) for key, item in value.items()}


def _current_user_namespace() -> Mapping[str, Any]:
    shell = get_ipython()
    if shell is None:
        return {}
    return cast(Mapping[str, Any], shell.user_ns)


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"agent context {name} must be a non-empty string")
    return value


def _metrics_from_payload(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError("agent context metrics must be an object")
    metrics: dict[str, float] = {}
    for name, metric_value in value.items():
        if not isinstance(name, str) or isinstance(metric_value, bool) or not isinstance(
            metric_value, (int, float)
        ):
            raise ValueError("agent context metrics must map string names to numbers")
        metrics[name] = float(metric_value)
    return metrics


def _evidence_from_payload(value: object) -> tuple[LLMEvidenceItem, ...]:
    if not isinstance(value, list):
        raise ValueError("agent context evidence must be an array")
    items: list[LLMEvidenceItem] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            raise ValueError("agent context evidence entries must be objects")
        evidence_id = _required_text(raw.get("evidence_id"), "evidence_id")
        detail = _required_text(raw.get("detail"), "evidence detail")
        items.append(LLMEvidenceItem(evidence_id, detail))
    return tuple(items)


def _strings_from_payload(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"agent context {name} must be an array of strings")
    return tuple(value)
