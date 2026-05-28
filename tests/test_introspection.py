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
