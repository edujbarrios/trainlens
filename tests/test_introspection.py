from trainlens.introspection import NotebookInspector


class DemoModel:
    def fit(self, x, y):
        return self

    def predict(self, x):
        return [0 for _ in x]


def test_inspector_detects_model_like_object():
    inspector = NotebookInspector()
    snapshot = inspector.snapshot({"model": DemoModel(), "_private": 1})

    candidates = inspector.find_models(snapshot)

    assert candidates
    assert candidates[0].variable_name == "model"
    assert snapshot.by_name("_private") is None


def test_inspector_ignores_helper_classes_and_functions():
    class HelperModelClass:
        def fit(self, x, y):
            return self

    def helper_function():
        return None

    inspector = NotebookInspector()
    snapshot = inspector.snapshot(
        {
            "model": DemoModel(),
            "HelperModelClass": HelperModelClass,
            "helper_function": helper_function,
        }
    )

    candidates = inspector.find_models(snapshot)

    assert [candidate.variable_name for candidate in candidates] == ["model"]
    assert snapshot.by_name("HelperModelClass") is None
    assert snapshot.by_name("helper_function") is None
