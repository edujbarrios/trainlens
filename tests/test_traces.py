from trainlens.analyzers.traces import extract_trace_events
from trainlens.pipeline import explain_namespace
from trainlens.renderers.markdown import MarkdownRenderer


def test_extracts_recent_execution_trace_events():
    events = extract_trace_events(
        {
            "training_trace": [
                {"step": 1, "epoch": 0.1, "event": "batch_end", "loss": 2.5},
                {"step": 2, "epoch": 0.2, "event": "eval", "eval_loss": 2.1},
            ]
        }
    )

    assert len(events) == 2
    assert events[0].step == 1
    assert events[1].name == "eval"
    assert events[1].metrics == {"eval_loss": 2.1}


def test_pipeline_renders_execution_trace_from_log_history():
    result = explain_namespace(
        {
            "log_history": [
                {"step": 10, "loss": 1.8},
                {"step": 20, "epoch": 1.0, "eval_loss": 1.4},
            ]
        }
    )

    markdown = MarkdownRenderer().render(result)

    assert result.trace
    assert "### Execution trace" in markdown
    assert "| 20 | 1.00 | training event | eval_loss=1.400 |" in markdown
