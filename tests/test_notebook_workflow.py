from __future__ import annotations

from types import SimpleNamespace

import pytest

from trainlens import LiveReport, preview_notebook_context
from trainlens.introspection import NotebookInspector
from trainlens.llm.context import build_llm_notebook_context
from trainlens.magic.commands import TrainLensMagics
from trainlens.models.analysis import AnalysisResult
from trainlens.pipeline import explain_namespace
from trainlens.storage.memory import InMemoryRunStore


class DemoShell:
    def __init__(self) -> None:
        self.user_ns = {
            "history": {
                "train_loss": [1.0, 0.7, 0.5],
                "val_loss": [1.1, 0.8, 0.6],
            }
        }


class FakeParameter:
    requires_grad = True

    def numel(self) -> int:
        return 10


class FakeTorchModel:
    def parameters(self):
        return iter((FakeParameter(),))

    def state_dict(self) -> dict[str, object]:
        return {}

    def train(self) -> None:
        return None


FakeTorchModel.__module__ = "torch.nn.modules.module"


def test_live_report_exposes_native_markdown_representation() -> None:
    result = AnalysisResult(model_name="demo")
    report = LiveReport(result=result, markdown="## Rendered\n")

    assert report._repr_markdown_() == "## Rendered\n"
    assert report.result is result
    assert report.markdown == "## Rendered\n"


def test_preview_uses_exact_context_builder_without_calling_provider(monkeypatch) -> None:
    def fail_provider(*args, **kwargs):
        raise AssertionError("provider must not be called during preview")

    monkeypatch.setattr("trainlens.notebook.explain_with_llm", fail_provider)
    namespace = {"history": {"loss": [1.0, 0.8]}}

    preview = preview_notebook_context(namespace)
    direct = build_llm_notebook_context(namespace)

    assert preview.markdown == direct.markdown
    assert preview.metrics == direct.metrics
    assert preview._repr_markdown_() == direct.markdown


def test_magic_dry_run_makes_no_provider_call_and_does_not_capture(monkeypatch) -> None:
    magics = TrainLensMagics(DemoShell())
    displayed: list[object] = []

    def fail_provider(*args, **kwargs):
        raise AssertionError("provider must not be called during dry-run")

    monkeypatch.setattr("trainlens.magic.commands.explain_with_llm", fail_provider)
    monkeypatch.setattr("trainlens.magic.commands.display", displayed.append)

    magics.explain_training("--dry-run")

    assert magics.store.runs == ()
    assert displayed
    assert "TrainLens Notebook Context" in displayed[0].data


def test_magic_captures_full_local_result_before_provider_failure(monkeypatch) -> None:
    shell = DemoShell()
    shell.user_ns["model"] = FakeTorchModel()
    expected = explain_namespace(shell.user_ns)
    magics = TrainLensMagics(shell)
    displayed: list[object] = []

    def fail_provider(*args, **kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr("trainlens.magic.commands.explain_with_llm", fail_provider)
    monkeypatch.setattr("trainlens.magic.commands.display", displayed.append)

    magics.explain_training("--name baseline")

    captured = magics.store.latest()
    assert captured is not None
    assert captured.model_name == expected.model_name
    assert captured.framework == expected.framework == "pytorch"
    assert captured.signals == expected.signals
    assert captured.recommendations == expected.recommendations
    assert magics.store.names == ("baseline",)
    assert "captured locally" in displayed[0].data


def test_magic_no_llm_supports_named_capture_and_explicit_comparison(monkeypatch) -> None:
    shell = DemoShell()
    magics = TrainLensMagics(shell)
    displayed: list[object] = []

    def fail_provider(*args, **kwargs):
        raise AssertionError("--no-llm must not call provider")

    monkeypatch.setattr("trainlens.magic.commands.explain_with_llm", fail_provider)
    monkeypatch.setattr("trainlens.magic.commands.display", displayed.append)

    magics.explain_training("--name baseline --no-llm")
    shell.user_ns["history"] = {
        "train_loss": [1.0, 0.6, 0.4],
        "val_loss": [1.1, 0.7, 0.5],
    }
    magics.explain_training("--name experiment --no-llm")
    magics.compare_runs("baseline experiment")

    comparison = displayed[-1].data
    assert "**Baseline:** baseline" in comparison
    assert "**Experiment:** experiment" in comparison
    assert "validation_loss" in comparison


def test_run_store_allows_names_and_one_based_indices() -> None:
    store = InMemoryRunStore()
    store.capture(AnalysisResult(model_name="A", metrics={"loss": 1.0}), name="baseline")
    store.capture(AnalysisResult(model_name="B", metrics={"loss": 0.8}), name="candidate")

    assert store.get("baseline").model_name == "A"
    assert store.get(2).model_name == "B"
    assert "**Baseline:** baseline" in store.render_comparison("baseline", "candidate")
    assert "**Experiment:** candidate" in store.render_comparison(1, 2)

    with pytest.raises(ValueError, match="unknown run"):
        store.get("missing")
    with pytest.raises(ValueError, match="already captured"):
        store.capture(AnalysisResult(), name="baseline")


def test_llm_context_includes_irregular_steps_and_fractional_epochs() -> None:
    context = build_llm_notebook_context(
        {
            "eval_log": [
                {"step": 100, "eval_loss": 0.9},
                {"step": 500, "eval_loss": 0.7},
                {"step": 1000, "eval_loss": 0.6},
            ],
            "epoch_log": [
                {"epoch": 0.25, "loss": 1.2},
                {"epoch": 0.5, "loss": 0.9},
            ],
        }
    )

    assert "`validation_loss`: points=[(100, 0.9), (500, 0.7), (1000, 0.6)]" in context.markdown
    assert "`loss`: points=[(0.25, 1.2), (0.5, 0.9)]" in context.markdown


def test_long_metric_point_history_remains_bounded() -> None:
    log = [{"step": step * 10, "eval_loss": 1.0 / step} for step in range(1, 21)]

    context = build_llm_notebook_context({"eval_log": log}, max_metric_points=4)
    metric_line = next(
        line
        for line in context.markdown.splitlines()
        if line.startswith("- `validation_loss`")
    )

    assert "observations=20" in metric_line
    assert "first_step=10" in metric_line
    assert "last_step=200" in metric_line
    assert metric_line.count("(") == 4


def test_pytorch_model_candidate_is_deduplicated_with_merged_evidence() -> None:
    model = FakeTorchModel()
    inspector = NotebookInspector()
    snapshot = inspector.snapshot({"model": model})

    candidates = inspector.find_models(snapshot)

    assert len(candidates) == 1
    assert candidates[0].object_ref is model
    assert candidates[0].framework == "pytorch"
    assert any("parameters" in reason.lower() for reason in candidates[0].reasons)

    context = build_llm_notebook_context({"model": model})
    model_candidates = context.markdown.split("## Model Candidates", maxsplit=1)[1]
    model_lines = [
        line for line in model_candidates.splitlines() if line.startswith("- `model`")
    ]
    assert len(model_lines) == 1


def test_distinct_model_objects_remain_distinct_candidates() -> None:
    first = FakeTorchModel()
    second = FakeTorchModel()
    inspector = NotebookInspector()
    snapshot = inspector.snapshot({"first": first, "second": second})

    candidates = inspector.find_models(snapshot)

    assert {candidate.object_ref for candidate in candidates} == {first, second}


def test_explain_argument_errors_are_helpful() -> None:
    magics = TrainLensMagics(SimpleNamespace(user_ns={}))

    with pytest.raises(ValueError, match="Unknown argument"):
        magics.explain_training("--wat")
    with pytest.raises(ValueError, match="Usage: %compare_runs"):
        magics.compare_runs("only-one")
