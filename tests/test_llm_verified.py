from __future__ import annotations

import json
import re

import pytest

from trainlens.llm.verified import (
    LLMEvidenceItem,
    build_verified_improvement_plan,
    evidence_catalog,
    parse_verified_improvement_plan,
)
from trainlens.models.analysis import AnalysisResult, EvidenceRef, Signal


class FakeStructuredProvider:
    def explain(self, markdown_report: str, *, mode="paper_report", prompt_options=None) -> str:
        catalog = markdown_report.split("## Evidence IDs", maxsplit=1)[1]
        match = re.search(r"`([^`]+)`", catalog)
        evidence_id = match.group(1) if match is not None else "missing"
        payload = {
            "recommendations": [
                {
                    "action": "Run one controlled follow-up experiment.",
                    "rationale": (
                        "Use deterministic TrainLens evidence before changing more variables."
                    ),
                    "evidence_ids": [evidence_id],
                    "confidence": 0.8,
                    "success_criterion": (
                        "Improve the cited metric without a new material regression."
                    ),
                }
            ]
        }
        return "## TrainLens Verified Improvement Plan\n```json\n" + json.dumps(payload) + "\n```"


def test_evidence_catalog_has_stable_metric_signal_and_reference_ids() -> None:
    result = AnalysisResult(
        summary=["Validation loss stopped improving."],
        metrics={"validation_loss": 0.42},
        signals=[
            Signal(
                title="Validation drift",
                detail="Validation loss rose late in training.",
                evidence_refs=(
                    EvidenceRef(
                        source="history",
                        detail="late validation loss increase",
                        metric="validation_loss",
                        start_step=8,
                        end_step=10,
                    ),
                ),
            )
        ],
    )

    catalog = evidence_catalog(result)
    assert [item.evidence_id for item in catalog] == [
        "summary:1",
        "metric:validation_loss",
        "signal:1",
        "signal:1:evidence:1",
    ]


def test_parser_flags_evidence_ids_that_were_not_supplied() -> None:
    evidence = (LLMEvidenceItem("metric:f1", "f1=0.82"),)
    payload = {
        "recommendations": [
            {
                "action": "Try A",
                "rationale": "Because B",
                "evidence_ids": ["metric:f1", "invented:1"],
                "confidence": 0.7,
                "success_criterion": "f1 improves",
            }
        ]
    }
    response = f"```json\n{json.dumps(payload)}\n```"

    plan = parse_verified_improvement_plan(response, evidence=evidence)

    assert plan.unsupported_evidence_ids == ("invented:1",)
    assert not plan.is_fully_supported
    assert plan.recommendations[0].unsupported_evidence_ids == ("invented:1",)


def test_parser_rejects_invalid_structured_fields() -> None:
    evidence = (LLMEvidenceItem("metric:f1", "f1=0.82"),)
    payload = {
        "recommendations": [
            {
                "action": "Try A",
                "rationale": "Because B",
                "evidence_ids": ["metric:f1"],
                "confidence": 2,
                "success_criterion": "f1 improves",
            }
        ]
    }

    with pytest.raises(ValueError, match="between 0 and 1"):
        parse_verified_improvement_plan(json.dumps(payload), evidence=evidence)


def test_build_verified_plan_uses_real_deterministic_evidence_catalog() -> None:
    namespace = {
        "history": {
            "loss": [1.0, 0.8, 0.7],
            "val_loss": [1.1, 0.9, 0.95],
        }
    }

    plan = build_verified_improvement_plan(namespace, provider=FakeStructuredProvider())

    assert plan.recommendations
    assert plan.recommendations[0].evidence_ids
    assert plan.is_fully_supported
