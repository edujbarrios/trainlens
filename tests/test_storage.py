import pytest

from trainlens.models.analysis import AnalysisResult
from trainlens.storage.memory import InMemoryRunStore


def test_run_store_exposes_latest_and_clear():
    store = InMemoryRunStore()
    first = AnalysisResult(model_name="First")
    second = AnalysisResult(model_name="Second")

    store.capture(first)
    store.capture(second)

    assert store.latest() is second
    assert store.runs == (first, second)

    store.clear()

    assert store.latest() is None
    assert store.runs == ()


def test_run_store_respects_max_runs():
    store = InMemoryRunStore(max_runs=2)

    store.capture(AnalysisResult(model_name="First"))
    store.capture(AnalysisResult(model_name="Second"))
    store.capture(AnalysisResult(model_name="Third"))

    assert [run.model_name for run in store.runs] == ["Second", "Third"]


@pytest.mark.parametrize("max_runs", [True, 2.5, "2"])
def test_run_store_rejects_non_integer_limits(max_runs):
    with pytest.raises(TypeError, match="max_runs must be an integer or None"):
        InMemoryRunStore(max_runs=max_runs)


def test_run_store_renders_latest_pair_comparison():
    store = InMemoryRunStore()
    store.capture(AnalysisResult(model_name="Baseline", metrics={"validation_loss": 0.5}))
    store.capture(AnalysisResult(model_name="Experiment", metrics={"validation_loss": 0.4}))

    markdown = store.render_comparison()

    assert "**Baseline:** previous run" in markdown
    assert "**Experiment:** latest run" in markdown
    assert "validation_loss" in markdown
    assert "improved" in markdown
