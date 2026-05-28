"""Minimal TrainLens usage outside a notebook."""

from trainlens.pipeline import explain_namespace
from trainlens.renderers.markdown import MarkdownRenderer


class DemoModel:
    feature_importances_ = [0.35, 0.65]

    def predict(self, rows):
        return [1 for _ in rows]


namespace = {
    "model": DemoModel(),
    "history": {"accuracy": [0.81, 0.89], "val_accuracy": [0.79, 0.84]},
    "feature_names": ["age", "account_balance"],
    "y_train": [0, 0, 0, 1],
}

print(MarkdownRenderer().render(explain_namespace(namespace)))
