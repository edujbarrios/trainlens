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
