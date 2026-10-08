"""Guard the public installation contract against stale extra-install snippets."""

from __future__ import annotations

import json
from pathlib import Path


def test_documentation_uses_single_standard_install() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = [root / "README.md", *(root / "docs").rglob("*.md")]
    for path in paths:
        content = path.read_text(encoding="utf-8")
        assert 'pip install "trainlens[plots]"' not in content, path
        assert 'pip install "trainlens[pdf]"' not in content, path
        assert "0.8.x supports" not in content, path
        assert "PDF requires `trainlens[pdf]`" not in content, path


def test_quickstart_install_matches_readme() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "examples/quickstart.ipynb").read_text(encoding="utf-8"))
    cells = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert "%pip install -q trainlens" in cells
    assert not any("trainlens[plots]" in cell or "trainlens[pdf]" in cell for cell in cells)
