"""Keep the public Colab notebook runnable and isolated from later examples."""

import json
from pathlib import Path


def test_long_run_notebook_cell_compiles_without_training_dependencies() -> None:
    path = Path(__file__).resolve().parents[1] / "examples" / "quickstart.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    candidates = [
        cell
        for cell in notebook["cells"]
        if "long-run-plot-demo" in cell.get("metadata", {}).get("tags", [])
    ]
    assert len(candidates) == 1
    source = "".join(candidates[0]["source"])
    compile(source, str(path), "exec")
    assert "max_plot_points=300" in source
    assert "x_axis=\"step\"" in source
    assert "preview_notebook_context" in source
    # Avoid overwriting globals used by the optional LLM example downstream.
    assert source.startswith("def demo_long_training_plot():")
    assert candidates[0]["outputs"] == []
    assert candidates[0]["execution_count"] is None


def test_quickstart_uses_plain_install_and_no_plot_extras() -> None:
    path = Path(__file__).resolve().parents[1] / "examples" / "quickstart.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    installs = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "%pip install" in "".join(cell["source"])
    ]
    assert installs == ["%pip install -q trainlens"]
