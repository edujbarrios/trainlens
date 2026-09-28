from trainlens.introspection import NotebookInspector
from trainlens.magic.commands import TrainLensMagics


class DemoShell:
    def __init__(self) -> None:
        self.user_ns = {"history": {"loss": [1.0, 0.8, 0.6]}}


def test_explain_magic_captures_notebook_state_once(monkeypatch) -> None:
    calls = 0
    original_snapshot = NotebookInspector.snapshot

    def counting_snapshot(self, namespace):
        nonlocal calls
        calls += 1
        return original_snapshot(self, namespace)

    monkeypatch.setattr(NotebookInspector, "snapshot", counting_snapshot)
    monkeypatch.setattr("trainlens.magic.commands.display", lambda _value: None)

    TrainLensMagics(DemoShell()).explain_training("--no-llm")

    assert calls == 1
