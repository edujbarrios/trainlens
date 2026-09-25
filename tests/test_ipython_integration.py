from __future__ import annotations

from IPython.core.interactiveshell import InteractiveShell

from trainlens.magic.extension import load_ipython_extension, unload_ipython_extension


def test_real_ipython_extension_load_execute_unload_and_reload(monkeypatch) -> None:
    InteractiveShell.clear_instance()
    shell = InteractiveShell.instance()
    displayed: list[object] = []
    monkeypatch.setattr("trainlens.magic.commands.display", displayed.append)
    try:
        shell.user_ns["history"] = {
            "train_loss": [1.0, 0.8],
            "val_loss": [1.1, 0.9],
        }

        load_ipython_extension(shell)
        assert shell.find_line_magic("explain_training") is not None
        assert shell.find_line_magic("suggest_improvements") is not None
        assert shell.find_line_magic("compare_runs") is not None

        shell.run_line_magic("explain_training", "--name baseline --no-llm")
        shell.user_ns["history"] = {
            "train_loss": [1.0, 0.7],
            "val_loss": [1.1, 0.8],
        }
        shell.run_line_magic("explain_training", "--name candidate --no-llm")
        shell.run_line_magic("compare_runs", "baseline candidate")

        magics = getattr(shell, "_trainlens_magics")
        assert magics.store.names == ("baseline", "candidate")
        assert "**Baseline:** baseline" in displayed[-1].data

        unload_ipython_extension(shell)
        assert shell.find_line_magic("explain_training") is None
        assert shell.find_line_magic("suggest_improvements") is None
        assert shell.find_line_magic("compare_runs") is None

        load_ipython_extension(shell)
        assert shell.find_line_magic("explain_training") is not None
        assert getattr(shell, "_trainlens_magics") is magics
        assert magics.store.names == ("baseline", "candidate")

        unload_ipython_extension(shell)
        load_ipython_extension(shell)
        unload_ipython_extension(shell)
    finally:
        unload_ipython_extension(shell)
        InteractiveShell.clear_instance()
