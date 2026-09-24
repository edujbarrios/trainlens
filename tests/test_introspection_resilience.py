from trainlens.introspection import NotebookInspector


class BrokenHistoryObject:
    @property
    def history(self):
        raise RuntimeError("history is unavailable")


class BrokenModelProbe:
    @property
    def fit(self):
        raise RuntimeError("fit is unavailable")


class ValidModel:
    def fit(self, *_args, **_kwargs) -> None:
        return None


def test_snapshot_skips_failing_framework_probe_without_losing_other_state() -> None:
    namespace = {
        "history": {"loss": [1.0, 0.8]},
        "broken": BrokenHistoryObject(),
    }

    snapshot = NotebookInspector().snapshot(namespace)

    assert snapshot.raw_namespace["history"] == {"loss": [1.0, 0.8]}
    assert snapshot.raw_namespace["broken"] is namespace["broken"]
    assert {variable.name for variable in snapshot.variables} == {"history", "broken"}


def test_model_detection_skips_failing_object_and_keeps_valid_candidate() -> None:
    namespace = {
        "broken": BrokenModelProbe(),
        "model": ValidModel(),
    }
    inspector = NotebookInspector()

    candidates = inspector.find_models(inspector.snapshot(namespace))

    assert [candidate.variable_name for candidate in candidates] == ["model"]
