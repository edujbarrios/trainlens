from trainlens.pipeline import explain_namespace


class DemoModel:
    feature_importances_ = [0.2, 0.8]

    def fit(self, x, y):
        return self

    def predict(self, x):
        return [0 for _ in x]


def test_pipeline_generates_report_with_features():
    result = explain_namespace(
        {
            "model": DemoModel(),
            "history": {"accuracy": [0.7, 0.9], "val_accuracy": [0.68, 0.74]},
            "feature_names": ["age", "balance"],
            "y_train": [0, 0, 0, 1],
        }
    )

    assert result.model_name == "DemoModel"
    assert result.top_features[0] == "balance"
    assert result.recommendations


def test_pipeline_summarizes_loss_changes():
    result = explain_namespace(
        {
            "history": {
                "train_loss": [2.0, 1.6, 1.3],
                "eval_loss": [2.1, 1.8, 1.7],
            }
        }
    )

    assert "Training loss changed from 2.000 to 1.300." in result.summary
    assert "Validation loss changed from 2.100 to 1.700." in result.summary
