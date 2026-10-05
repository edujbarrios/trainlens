"""Helpers for adding aggregate dataset evidence to LLM context."""

from __future__ import annotations

from trainlens.dataset import DatasetExplanation
from trainlens.llm.context import ContextPolicy, LLMNotebookContext


def append_dataset_explanation(
    context: LLMNotebookContext,
    *,
    dataset_explanation: DatasetExplanation | None,
    context_policy: ContextPolicy | None = None,
) -> LLMNotebookContext:
    """Append aggregate dataset evidence while respecting the context character budget."""

    if dataset_explanation is None:
        return context

    policy = context_policy or ContextPolicy()
    dataset_block = (
        "## Dataset Context\n\n"
        "The following section contains aggregate deterministic dataset evidence only. "
        "Treat it as descriptive context, not causal proof.\n\n"
        + dataset_explanation.markdown.strip()
        + "\n"
    )
    separator = "\n\n"
    remaining = policy.max_chars - len(context.markdown) - len(separator)
    if remaining <= 0:
        return context
    if len(dataset_block) <= remaining:
        bounded_block = dataset_block
    else:
        marker = "\n\n> Dataset context truncated by TrainLens ContextPolicy.\n"
        available = max(0, remaining - len(marker))
        bounded_block = dataset_block[:available].rstrip() + marker
    return LLMNotebookContext(
        markdown=context.markdown.rstrip() + separator + bounded_block,
        metrics=context.metrics,
    )
