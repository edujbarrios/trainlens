from __future__ import annotations

import json
import math

import pytest

from trainlens import ExperimentRun, render_report, suggest_next_experiment, write_report
from trainlens.export import render_report as module_render_report
from trainlens.models.analysis import AnalysisResult, Recommendation, Signal
from trainlens.notebook import LiveReport


def _result() -> AnalysisResult:
    return AnalysisResult(
        model_name="LogisticRegression",
        framework="sklearn",
        metrics={"validation_accuracy": 0.91},
        signals=[Signal("Stable validation", "Validation accuracy improved.")],
        recommendations=[Recommendation("Inspect validation errors.", "Find failure modes.")],
    )


def test_render_report_exports_markdown_html_and_json() -> None:
    result = _result()

    markdown = render_report(result, format="markdown")
    html = render_report(result, format="html")
    payload = json.loads(str(render_report(result, format="json")))

    assert "LogisticRegression" in str(markdown)
    assert "<!doctype html>" in str(html)
    assert "<h2>TrainLens Report</h2>" in str(html)
    assert payload["model_name"] == "LogisticRegression"
    assert payload["recommendations"][0]["action"] == "Inspect validation errors."


def test_html_export_renders_markdown_inline_styles_and_ordered_lists() -> None:
    html = str(render_report(_result(), format="html"))

    assert "<strong>Model detected:</strong>" in html
    assert "<em>Find failure modes.</em>" in html
    assert "<ol>" in html
    assert "<li>Inspect validation errors. Confidence: 50%. Why: Find failure modes.</li>" in html


def test_render_report_exports_live_report_json() -> None:
    report = LiveReport(result=_result(), markdown="## Custom LLM Report\n")

    payload = json.loads(str(module_render_report(report, format="json")))

    assert payload["markdown"] == "## Custom LLM Report\n"
    assert payload["result"]["framework"] == "sklearn"


def test_json_export_replaces_non_finite_values_with_null() -> None:
    result = AnalysisResult(metrics={"loss": math.nan, "gradient_norm": math.inf})

    rendered = str(render_report(result, format="json"))
    payload = json.loads(rendered, parse_constant=lambda value: pytest.fail(value))

    assert payload["metrics"] == {"gradient_norm": None, "loss": None}
    assert "NaN" not in rendered
    assert "Infinity" not in rendered


def test_render_report_exports_next_experiment_in_supported_formats() -> None:
    recommendation = suggest_next_experiment(
        [
            ExperimentRun(
                name="baseline",
                metrics={"validation_loss": 0.5},
                parameters={"learning_rate": 1e-3},
            )
        ]
    )

    markdown = str(render_report(recommendation, format="markdown"))
    html = str(render_report(recommendation, format="html"))
    payload = json.loads(str(render_report(recommendation, format="json")))

    assert "## TrainLens Next Experiment" in markdown
    assert "<h2>TrainLens Next Experiment</h2>" in html
    assert payload["source_run"] == "baseline"
    assert payload["success_criteria"][0]["metric"] == "validation_loss"


def test_write_report_infers_format_from_suffix(tmp_path) -> None:
    path = tmp_path / "report.md"

    written = write_report(_result(), path)

    assert written == path
    assert "TrainLens Report" in path.read_text(encoding="utf-8")


def test_pdf_export_explains_optional_dependency(monkeypatch) -> None:
    def fake_import_module(name: str) -> object:
        if name == "reportlab.pdfgen.canvas":
            raise ImportError
        raise AssertionError(name)

    monkeypatch.setattr("trainlens.export.import_module", fake_import_module)

    with pytest.raises(RuntimeError, match=r"trainlens\[pdf\]"):
        render_report(_result(), format="pdf")
