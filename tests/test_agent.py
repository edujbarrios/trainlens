from __future__ import annotations

import json

from trainlens import (
    AgentContext,
    TrainingRun,
    build_agent_context,
    build_agent_context_from_run,
    verify_agent_plan,
)
from trainlens.models.metric import MetricSeries


def _history() -> dict[str, object]:
    return {
        "train_loss": [1.0, 0.7, 0.45, 0.30],
        "val_loss": [1.0, 0.72, 0.61, 0.66],
    }


def test_build_agent_context_is_provider_free_and_serializable() -> None:
    context = build_agent_context(_history())

    assert context.objective == "propose_next_experiment"
    assert context.evidence
    assert "TrainLens made no LLM request" in context.markdown
    assert "Do not tune against held-out test evidence" in context.markdown

    restored = AgentContext.from_dict(json.loads(context.to_json()))
    assert restored.objective == context.objective
    assert restored.metrics == context.metrics
    assert restored.evidence == context.evidence


def test_verify_agent_plan_accepts_supported_evidence_and_flags_unknown_ids() -> None:
    context = build_agent_context(_history())
    evidence_id = context.evidence[0].evidence_id
    response = {
        "recommendations": [
            {
                "action": "shorten training or add early stopping",
                "rationale": "validation evidence degrades late in the run",
                "evidence_ids": [evidence_id],
                "confidence": 0.8,
                "success_criterion": "best validation loss improves without a wider gap",
            }
        ]
    }

    supported = verify_agent_plan(response, context=context)
    assert supported.is_fully_supported
    assert supported.unsupported_evidence_ids == ()

    response["recommendations"][0]["evidence_ids"] = ["invented:evidence"]
    unsupported = verify_agent_plan(response, context=context)
    assert not unsupported.is_fully_supported
    assert unsupported.unsupported_evidence_ids == ("invented:evidence",)


def test_build_agent_context_from_portable_run_exposes_recorded_parameters() -> None:
    run = TrainingRun(
        run_id="run-1",
        model_name="demo-model",
        framework="pytorch",
        metrics=(MetricSeries("validation_loss", (0.8, 0.6, 0.62)),),
        parameters={"learning_rate": 2e-5, "lora_r": 16},
        notes=("baseline adapter run",),
    )

    context = build_agent_context_from_run(run)
    evidence_ids = {item.evidence_id for item in context.evidence}

    assert "run:id" in evidence_ids
    assert "run:model" in evidence_ids
    assert "run:framework" in evidence_ids
    assert "run:parameter:learning_rate" in evidence_ids
    assert "run:parameter:lora_r" in evidence_ids
    assert "run:note:1" in evidence_ids
