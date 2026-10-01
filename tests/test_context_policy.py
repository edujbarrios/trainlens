import pytest

from trainlens import ContextPolicy, build_paper_report, preview_notebook_context
from trainlens.llm.context import build_llm_notebook_context


def test_context_policy_limits_notebook_variables_with_explicit_notice() -> None:
    context = build_llm_notebook_context(
        {"alpha": 1, "beta": 2, "gamma": 3, "delta": 4},
        context_policy=ContextPolicy(max_variables=2),
    )

    assert "- `alpha`: type=int" in context.markdown
    assert "- `beta`: type=int" in context.markdown
    assert "2 additional notebook variables omitted by ContextPolicy" in context.markdown


def test_context_policy_limits_rendered_metric_series_without_dropping_metric_results() -> None:
    context = build_llm_notebook_context(
        {"history": {"loss": [1.0, 0.5], "accuracy": [0.5, 0.8]}},
        context_policy=ContextPolicy(max_metric_series=1),
    )

    assert "1 additional metric series omitted by ContextPolicy" in context.markdown
    assert context.metrics == {"accuracy": 0.8, "loss": 0.5}


def test_context_policy_enforces_a_global_character_budget_with_notice() -> None:
    namespace = {f"value_{index}": "x" * 80 for index in range(30)}

    context = build_llm_notebook_context(
        namespace,
        include_values=True,
        context_policy=ContextPolicy(max_variables=30, max_chars=700),
    )

    assert len(context.markdown) <= 700
    assert "TrainLens Deterministic Findings" in context.markdown
    assert "context truncated by ContextPolicy" in context.markdown


def test_preview_notebook_context_accepts_context_policy() -> None:
    context = preview_notebook_context(
        {"alpha": 1, "beta": 2, "gamma": 3},
        context_policy=ContextPolicy(max_variables=1),
    )

    assert "2 additional notebook variables omitted by ContextPolicy" in context.markdown


def test_report_helpers_forward_context_policy_to_injected_provider() -> None:
    captured: dict[str, str] = {}

    class Provider:
        def explain(self, markdown_report: str, **_kwargs: object) -> str:
            captured["markdown"] = markdown_report
            return "## Report"

    report = build_paper_report(
        {"alpha": 1, "beta": 2, "gamma": 3},
        provider=Provider(),
        context_policy=ContextPolicy(max_variables=1),
    )

    assert report.markdown == "## Report"
    assert "2 additional notebook variables omitted by ContextPolicy" in captured["markdown"]


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"max_metric_points": 1}, ValueError, "max_metric_points"),
        ({"max_metric_series": 0}, ValueError, "max_metric_series"),
        ({"max_variables": True}, TypeError, "max_variables"),
        ({"max_chars": 100}, ValueError, "max_chars"),
    ],
)
def test_context_policy_rejects_invalid_budgets(kwargs, error_type, message) -> None:
    with pytest.raises(error_type, match=message):
        ContextPolicy(**kwargs)
