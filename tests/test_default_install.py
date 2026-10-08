"""The plain PyPI install must provide plotting and PDF support."""

from __future__ import annotations

import tomllib
from pathlib import Path

from trainlens import plot_training_curves, render_report
from trainlens.models.analysis import AnalysisResult


def test_base_dependencies_include_all_builtin_export_capabilities() -> None:
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert any(item.startswith("matplotlib>=") for item in project["dependencies"])
    assert any(item.startswith("reportlab>=") for item in project["dependencies"])
    assert project["optional-dependencies"]["plots"] == []
    assert project["optional-dependencies"]["pdf"] == []


def test_plotting_and_pdf_work_without_extras() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure = plot_training_curves(
        {"history": {"train_loss": [1.2, 0.8], "val_loss": [1.3, 0.9]}},
        metrics=("loss",),
    )
    assert len(figure.axes) == 1
    plt.close(figure)
    pdf = render_report(AnalysisResult(metrics={"loss": 0.5}), format="pdf")
    assert isinstance(pdf, bytes) and pdf.startswith(b"%PDF-")
