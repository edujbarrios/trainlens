"""Structured, evidence-verified LLM improvement plans."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from IPython import get_ipython

from trainlens.analysis_config import AnalysisConfig
from trainlens.dataset import DatasetExplanation, DatasetFeatureSummary, DatasetTargetSummary
from trainlens.llm.context import ContextPolicy, build_llm_notebook_context_from_snapshot
from trainlens.llm.dataset_context import append_dataset_explanation
from trainlens.llm.enhancer import explain_with_llm
from trainlens.llm.prompts import PromptOptions
from trainlens.llm.provider import LLMProvider
from trainlens.models.analysis import AnalysisResult
from trainlens.pipeline import analyze_snapshot, snapshot_namespace

_JSON_FENCE = re.compile(r"```json\s*(\{.*?\})\s*```", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class LLMEvidenceItem:
    """One deterministic TrainLens fact that an LLM may cite by stable ID."""

    evidence_id: str
    detail: str


@dataclass(frozen=True)
class VerifiedLLMRecommendation:
    """One parsed LLM recommendation with citation verification results."""

    action: str
    rationale: str
    evidence_ids: tuple[str, ...]
    confidence: float
    success_criterion: str
    unsupported_evidence_ids: tuple[str, ...] = ()

    @property
    def is_supported(self) -> bool:
        """Return whether every cited evidence ID exists in the supplied catalog."""

        return not self.unsupported_evidence_ids and bool(self.evidence_ids)


@dataclass(frozen=True)
class VerifiedImprovementPlan:
    """Machine-readable LLM plan checked against deterministic TrainLens evidence."""

    recommendations: tuple[VerifiedLLMRecommendation, ...]
    evidence: tuple[LLMEvidenceItem, ...]
    raw_response: str

    @property
    def unsupported_evidence_ids(self) -> tuple[str, ...]:
        """Return unique evidence IDs invented or mis-cited by the LLM."""

        invalid: list[str] = []
        for recommendation in self.recommendations:
            for evidence_id in recommendation.unsupported_evidence_ids:
                if evidence_id not in invalid:
                    invalid.append(evidence_id)
        return tuple(invalid)

    @property
    def is_fully_supported(self) -> bool:
        """Return whether every recommendation cites valid deterministic evidence."""

        return bool(self.recommendations) and all(
            recommendation.is_supported for recommendation in self.recommendations
        )


def evidence_catalog(
    result: AnalysisResult,
    *,
    dataset_explanation: DatasetExplanation | None = None,
) -> tuple[LLMEvidenceItem, ...]:
    """Build stable evidence IDs from deterministic model and dataset analysis."""

    items: list[LLMEvidenceItem] = []
    for index, summary in enumerate(result.summary, start=1):
        items.append(LLMEvidenceItem(f"summary:{index}", summary))
    for name, value in sorted(result.metrics.items()):
        items.append(LLMEvidenceItem(f"metric:{name}", f"{name}={value:.8g}"))
    for index, signal in enumerate(result.signals, start=1):
        items.append(
            LLMEvidenceItem(
                f"signal:{index}",
                f"{signal.title}: {signal.detail} (severity={signal.severity})",
            )
        )
        for evidence_index, evidence in enumerate(signal.evidence_refs, start=1):
            parts = [f"source={evidence.source}", f"detail={evidence.detail}"]
            if evidence.metric is not None:
                parts.append(f"metric={evidence.metric}")
            if evidence.start_step is not None:
                parts.append(f"start_step={evidence.start_step}")
            if evidence.end_step is not None:
                parts.append(f"end_step={evidence.end_step}")
            items.append(
                LLMEvidenceItem(
                    f"signal:{index}:evidence:{evidence_index}",
                    "; ".join(parts),
                )
            )
    if dataset_explanation is not None:
        items.extend(_dataset_evidence(dataset_explanation))
    return tuple(items)


def build_verified_improvement_plan(
    namespace: Mapping[str, Any] | None = None,
    *,
    provider: LLMProvider | None = None,
    analysis_config: AnalysisConfig | None = None,
    context_policy: ContextPolicy | None = None,
    max_metric_points: int = 12,
    include_values: bool = False,
    dataset_explanation: DatasetExplanation | None = None,
) -> VerifiedImprovementPlan:
    """Generate a structured improvement plan and verify every evidence citation."""

    report_namespace = _current_user_namespace() if namespace is None else namespace
    snapshot = snapshot_namespace(report_namespace)
    result = analyze_snapshot(snapshot, config=analysis_config)
    context = build_llm_notebook_context_from_snapshot(
        snapshot,
        max_metric_points=max_metric_points,
        include_values=include_values,
        analysis_config=analysis_config,
        deterministic_result=result,
        context_policy=context_policy,
    )
    context = append_dataset_explanation(
        context,
        dataset_explanation=dataset_explanation,
        context_policy=context_policy,
    )
    evidence = evidence_catalog(result, dataset_explanation=dataset_explanation)
    evidence_markdown = _render_evidence_catalog(evidence)
    prompt_options = PromptOptions(
        objective=(
            "Produce a machine-readable improvement plan grounded only in the supplied "
            "TrainLens evidence IDs."
        ),
        heading="## TrainLens Verified Improvement Plan",
        return_instructions=(
            "After the heading, return exactly one fenced `json` object and no prose after it.",
            "The JSON root must be an object with a `recommendations` array.",
            "Each recommendation must contain `action`, `rationale`, `evidence_ids`, "
            "`confidence`, and `success_criterion`.",
            "`evidence_ids` must contain only IDs from the `Evidence IDs` section.",
            "`confidence` must be a number from 0 to 1.",
        ),
        focus_areas=(
            "highest-information next experiment",
            "controlled changes",
            "measurable success criteria",
        ),
    )
    response = explain_with_llm(
        context.markdown + "\n\n" + evidence_markdown,
        mode="improvement_ideas",
        require_provider=True,
        prompt_options=prompt_options,
        provider=provider,
    )
    return parse_verified_improvement_plan(response, evidence=evidence)


def parse_verified_improvement_plan(
    response: str,
    *,
    evidence: tuple[LLMEvidenceItem, ...],
) -> VerifiedImprovementPlan:
    """Parse one structured LLM response and flag unsupported evidence references."""

    payload = _extract_json_object(response)
    recommendations_raw = payload.get("recommendations")
    if not isinstance(recommendations_raw, list):
        raise ValueError("verified improvement plan must contain a recommendations array")

    valid_ids = {item.evidence_id for item in evidence}
    recommendations: list[VerifiedLLMRecommendation] = []
    for index, raw in enumerate(recommendations_raw, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"recommendation {index} must be an object")
        action = _required_string(raw, "action", index)
        rationale = _required_string(raw, "rationale", index)
        success_criterion = _required_string(raw, "success_criterion", index)
        evidence_ids = _evidence_ids(raw.get("evidence_ids"), index)
        confidence = _confidence(raw.get("confidence"), index)
        unsupported = tuple(item for item in evidence_ids if item not in valid_ids)
        recommendations.append(
            VerifiedLLMRecommendation(
                action=action,
                rationale=rationale,
                evidence_ids=evidence_ids,
                confidence=confidence,
                success_criterion=success_criterion,
                unsupported_evidence_ids=unsupported,
            )
        )
    if not recommendations:
        raise ValueError("verified improvement plan must contain at least one recommendation")
    return VerifiedImprovementPlan(tuple(recommendations), evidence, response)


def _dataset_evidence(explanation: DatasetExplanation) -> tuple[LLMEvidenceItem, ...]:
    items: list[LLMEvidenceItem] = []
    for split_index, split in enumerate(explanation.splits, start=1):
        prefix = f"dataset:split:{split_index}"
        items.append(
            LLMEvidenceItem(
                f"{prefix}:rows",
                f"split={split.name}; rows={split.row_count}",
            )
        )
        for feature_index, feature in enumerate(split.features, start=1):
            items.append(
                LLMEvidenceItem(
                    f"{prefix}:feature:{feature_index}",
                    _feature_evidence_detail(split.name, feature),
                )
            )
        if split.target is not None:
            items.append(
                LLMEvidenceItem(
                    f"{prefix}:target",
                    _target_evidence_detail(split.name, split.target),
                )
            )
        for observation_index, observation in enumerate(split.observations, start=1):
            items.append(
                LLMEvidenceItem(
                    f"{prefix}:observation:{observation_index}",
                    observation,
                )
            )
    for observation_index, observation in enumerate(explanation.observations, start=1):
        items.append(
            LLMEvidenceItem(
                f"dataset:observation:{observation_index}",
                observation,
            )
        )
    return tuple(items)


def _feature_evidence_detail(split_name: str, feature: DatasetFeatureSummary) -> str:
    parts = [
        f"split={split_name}",
        f"feature={feature.name}",
        f"kind={feature.kind}",
        f"missing={feature.missing}/{feature.count}",
    ]
    if feature.unique is not None:
        parts.append(f"unique={feature.unique}")
    if feature.minimum is not None and feature.maximum is not None:
        parts.append(f"range={feature.minimum:.8g}..{feature.maximum:.8g}")
    if feature.mean is not None:
        parts.append(f"mean={feature.mean:.8g}")
    if feature.average_length is not None:
        parts.append(f"average_length={feature.average_length:.8g}")
    return "; ".join(parts)


def _target_evidence_detail(split_name: str, target: DatasetTargetSummary) -> str:
    parts = [
        f"split={split_name}",
        f"target={target.name}",
        f"kind={target.kind}",
        f"missing={target.missing}/{target.count}",
    ]
    if target.unique is not None:
        parts.append(f"unique={target.unique}")
    if target.minimum is not None and target.maximum is not None:
        parts.append(f"range={target.minimum:.8g}..{target.maximum:.8g}")
    if target.mean is not None:
        parts.append(f"mean={target.mean:.8g}")
    if target.classes:
        distribution = ", ".join(
            f"{item.label}:{item.fraction:.1%}" for item in target.classes
        )
        parts.append(f"class_distribution={distribution}")
    return "; ".join(parts)


def _render_evidence_catalog(evidence: tuple[LLMEvidenceItem, ...]) -> str:
    lines = ["## Evidence IDs"]
    if not evidence:
        lines.append("- No deterministic evidence IDs are available.")
    else:
        lines.extend(f"- `{item.evidence_id}`: {item.detail}" for item in evidence)
    return "\n".join(lines)


def _extract_json_object(response: str) -> dict[str, Any]:
    match = _JSON_FENCE.search(response)
    candidate = match.group(1) if match is not None else response.strip()
    if match is None:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]
    try:
        payload: object = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM response did not contain a valid JSON improvement plan") from exc
    if not isinstance(payload, dict):
        raise ValueError("verified improvement plan JSON root must be an object")
    return payload


def _required_string(raw: dict[str, Any], key: str, index: int) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"recommendation {index} field {key!r} must be a non-empty string")
    return value.strip()


def _evidence_ids(value: object, index: int) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"recommendation {index} evidence_ids must be an array of strings")
    return tuple(dict.fromkeys(item.strip() for item in value if item.strip()))


def _confidence(value: object, index: int) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"recommendation {index} confidence must be numeric")
    numeric = float(value)
    if not 0.0 <= numeric <= 1.0:
        raise ValueError(f"recommendation {index} confidence must be between 0 and 1")
    return numeric


def _current_user_namespace() -> Mapping[str, Any]:
    shell = get_ipython()
    if shell is None:
        raise RuntimeError("No active IPython shell found; pass a namespace explicitly.")
    return cast(Mapping[str, Any], shell.user_ns)
