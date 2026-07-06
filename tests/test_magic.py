from trainlens.magic.commands import TrainLensMagics


class DemoShell:
    user_ns = {
        "history": {"train_loss": [2.0, 1.5], "eval_loss": [2.1, 1.8]},
        "training_trace": [{"step": 1, "loss": 2.0}, {"step": 2, "eval_loss": 1.8}],
    }


def test_explain_training_magic_captures_run(monkeypatch):
    magics = TrainLensMagics(DemoShell())

    def fake_explain(
        markdown: str,
        *,
        mode: str = "paper_report",
        require_provider: bool = False,
    ) -> str:
        assert mode == "paper_report"
        assert require_provider is True
        assert "TrainLens Notebook Context" in markdown
        return "## TrainLens Report\n\nLLM magic report."

    monkeypatch.setattr("trainlens.magic.commands.explain_with_llm", fake_explain)

    magics.explain_training("")

    assert magics.store.latest() is not None


def test_suggest_improvements_magic_uses_improvement_mode(monkeypatch):
    magics = TrainLensMagics(DemoShell())
    captured: dict[str, object] = {}

    def fake_explain(
        markdown: str,
        *,
        mode: str = "paper_report",
        require_provider: bool = False,
    ) -> str:
        captured["mode"] = mode
        captured["require_provider"] = require_provider
        assert "TrainLens Notebook Context" in markdown
        return "## TrainLens Improvement Ideas"

    monkeypatch.setattr("trainlens.magic.commands.explain_with_llm", fake_explain)

    magics.suggest_improvements("")

    assert captured["mode"] == "improvement_ideas"
    assert captured["require_provider"] is True
