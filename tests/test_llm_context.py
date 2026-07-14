from trainlens.llm.context import build_llm_notebook_context


def test_llm_context_preserves_short_metric_series() -> None:
    context = build_llm_notebook_context({"history": {"loss": [1.0, 0.7, 0.4]}})

    assert "- `loss`: [1, 0.7, 0.4]" in context.markdown


def test_llm_context_compacts_long_metric_series_with_training_evidence() -> None:
    loss = [1.0 - index / 100 for index in range(50)]

    context = build_llm_notebook_context({"history": {"loss": loss}})

    metric_line = next(
        line for line in context.markdown.splitlines() if line.startswith("- `loss`")
    )
    assert "observations=50" in metric_line
    assert "first=1" in metric_line
    assert "last=0.51" in metric_line
    assert "min=0.51" in metric_line
    assert "max=1" in metric_line
    assert "ordered_sample=[" in metric_line
    assert metric_line.count(",") < 20
    assert context.metrics["loss"] == 0.51


def test_llm_context_does_not_repeat_metric_container_values() -> None:
    context = build_llm_notebook_context(
        {
            "history": {"loss": [1.0, 0.7, 0.4]},
            "dataset_name": "ag_news",
        }
    )

    assert "- `history`: type=dict" in context.markdown
    assert "value: {'loss': [1.0, 0.7, 0.4]}" not in context.markdown
    assert "- `loss`: [1, 0.7, 0.4]" in context.markdown
    assert "value: 'ag_news'" in context.markdown
