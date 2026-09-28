from trainlens.introspection import NotebookInspector
from trainlens.pipeline import explain_namespace


class BrokenHistoryObject:
    @property
    def history(self):
        raise RuntimeError("history is unavailable")


class BrokenModelProbe:
    @property
    def fit(self):
        raise RuntimeError("fit is unavailable")


class BrokenArrayLike:
    @property
    def shape(self):
        raise RuntimeError("shape is unavailable")


class ValidModel:
    def fit(self, *_args, **_kwargs) -> None:
        return None


class FragileModel:
    def fit(self, *_args, **_kwargs) -> None:
        return None

    @property
    def config(self):
        raise RuntimeError("config is unavailable")

    @property
    def feature_importances_(self):
        raise RuntimeError("feature importances are unavailable")

    @property
    def coef_(self):
        raise RuntimeError("coefficients are unavailable")


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


def test_analysis_ignores_failing_metric_shape_property() -> None:
    result = explain_namespace(
        {
            "loss_history": BrokenArrayLike(),
            "history": {"loss": [1.0, 0.8]},
        }
    )

    assert result.metrics["train_loss"] == 0.8


def test_analysis_ignores_failing_model_metadata_and_feature_properties() -> None:
    result = explain_namespace(
        {
            "model": FragileModel(),
            "history": {"loss": [1.0, 0.8]},
            "feature_names": ["a", "b"],
        }
    )

    assert result.model_name is not None
    assert result.metrics["train_loss"] == 0.8
    assert result.top_features == []
