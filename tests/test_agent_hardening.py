"""Regression tests for Agent Mode context privacy, IDs, and size limits."""

from types import SimpleNamespace

from trainlens import ContextPolicy, TrainingRun, build_agent_context_from_run
from trainlens.llm.context import LLMNotebookContext
from trainlens.llm.dataset_context import append_dataset_explanation


def test_portable_run_context_redacts_secrets() -> None:
    secret = "sk-abcdefghijklmnopqrstuvwxyz"
    run = TrainingRun(
        run_id="run-1",
        parameters={"api_key": secret, "comment": f"Bearer {secret}"},
        notes=(f"Authentication: Bearer {secret}",),
    )
    context = build_agent_context_from_run(run)
    assert secret not in context.to_json()
    assert "[REDACTED]" in context.to_json()
    assert any(item.evidence_id == "run:parameter:api_key" for item in context.evidence)


def test_parameter_evidence_ids_do_not_collide_after_normalization() -> None:
    run = TrainingRun(
        run_id="run-1",
        parameters={"learning rate": 0.01, "learning_rate": 0.02, "": 7},
    )
    context = build_agent_context_from_run(run)
    ids = [item.evidence_id for item in context.evidence]
    assert len(ids) == len(set(ids))
    assert "run:parameter:learning%20rate" in ids
    assert "run:parameter:learning_rate" in ids


def test_agent_markdown_respects_character_budget_and_only_verifies_visible_evidence() -> None:
    run = TrainingRun(
        run_id="run-1",
        parameters={f"p{index}": index for index in range(80)},
    )
    policy = ContextPolicy(max_chars=500)
    context = build_agent_context_from_run(run, context_policy=policy)
    assert len(context.markdown) <= policy.max_chars
    assert "shortened" in context.markdown
    assert all(f"`{item.evidence_id}`" in context.markdown for item in context.evidence)
    assert len(context.evidence) < 81


def test_dataset_truncation_marker_never_exceeds_remaining_budget() -> None:
    policy = ContextPolicy(max_chars=256)
    context = LLMNotebookContext(markdown="x" * 220, metrics={})
    explanation = SimpleNamespace(markdown="sample" * 1000)
    result = append_dataset_explanation(
        context, dataset_explanation=explanation, context_policy=policy
    )
    assert len(result.markdown) <= policy.max_chars
