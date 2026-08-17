"""Report export helpers."""

from __future__ import annotations

import html
import json
import re
import textwrap
from collections.abc import Mapping
from dataclasses import asdict
from importlib import import_module
from pathlib import Path
from typing import Any, Literal, Never, TypeAlias

from trainlens.comparison import render_run_comparison
from trainlens.experiments import NextExperimentRecommendation, render_next_experiment
from trainlens.models.analysis import AnalysisResult
from trainlens.models.comparison import RunComparison
from trainlens.notebook import LiveReport
from trainlens.renderers.markdown import MarkdownRenderer

ReportFormat: TypeAlias = Literal["markdown", "html", "json", "pdf"]
ReportContent: TypeAlias = str | bytes
ReportInput: TypeAlias = (
    AnalysisResult | LiveReport | RunComparison | NextExperimentRecommendation
)


def render_report(report: ReportInput, *, format: ReportFormat = "markdown") -> ReportContent:
    """Render a TrainLens report as Markdown, HTML, JSON, or PDF bytes."""

    markdown = _report_markdown(report)
    if format == "markdown":
        return markdown
    if format == "html":
        return _markdown_to_html(markdown)
    if format == "json":
        return _report_json(report)
    if format == "pdf":
        return _markdown_to_pdf(markdown)
    _assert_never(format)


def write_report(
    report: ReportInput,
    path: str | Path,
    *,
    format: ReportFormat | None = None,
) -> Path:
    """Write a TrainLens report to disk and return the output path."""

    output_path = Path(path)
    selected_format = format or _format_from_suffix(output_path)
    content = render_report(report, format=selected_format)
    if isinstance(content, bytes):
        output_path.write_bytes(content)
    else:
        output_path.write_text(content, encoding="utf-8")
    return output_path


def _report_markdown(report: ReportInput) -> str:
    if isinstance(report, LiveReport):
        return report.markdown
    if isinstance(report, RunComparison):
        return render_run_comparison(report)
    if isinstance(report, NextExperimentRecommendation):
        return render_next_experiment(report)
    return MarkdownRenderer().render(report)


def _report_json(report: ReportInput) -> str:
    payload: Mapping[str, Any]
    if isinstance(report, LiveReport):
        payload = {
            "markdown": report.markdown,
            "result": asdict(report.result),
        }
    else:
        payload = asdict(report)
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _markdown_to_html(markdown: str) -> str:
    body = "\n".join(_markdown_lines_to_html(markdown.splitlines()))
    return textwrap.dedent(
        f"""\
        <!doctype html>
        <html lang="en">
        <head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <title>TrainLens Report</title>
          <style>
            body {{ font-family: system-ui, sans-serif; line-height: 1.55; margin: 2rem; }}
            article {{ max-width: 960px; margin: 0 auto; }}
            table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
            th, td {{ border: 1px solid #d0d7de; padding: 0.45rem 0.6rem; }}
            th {{ background: #f6f8fa; text-align: left; }}
            code {{ background: #f6f8fa; padding: 0.1rem 0.25rem; border-radius: 4px; }}
          </style>
        </head>
        <body>
        <article>
        {body}
        </article>
        </body>
        </html>
        """
    )


def _markdown_lines_to_html(lines: list[str]) -> list[str]:
    rendered: list[str] = []
    in_list = False
    in_ordered_list = False
    in_table = False
    for line in lines:
        if in_table and not line.startswith("|"):
            rendered.append("</tbody></table>")
            in_table = False
        if in_list and not line.startswith("- "):
            rendered.append("</ul>")
            in_list = False
        if in_ordered_list and not _ordered_list_item(line):
            rendered.append("</ol>")
            in_ordered_list = False
        if not line.strip():
            continue
        if line.startswith("### "):
            rendered.append(f"<h3>{_inline_html(line[4:])}</h3>")
        elif line.startswith("## "):
            rendered.append(f"<h2>{_inline_html(line[3:])}</h2>")
        elif line.startswith("# "):
            rendered.append(f"<h1>{_inline_html(line[2:])}</h1>")
        elif line.startswith("- "):
            if not in_list:
                rendered.append("<ul>")
                in_list = True
            rendered.append(f"<li>{_inline_html(line[2:])}</li>")
        elif item := _ordered_list_item(line):
            if not in_ordered_list:
                rendered.append("<ol>")
                in_ordered_list = True
            rendered.append(f"<li>{_inline_html(item)}</li>")
        elif line.startswith("|"):
            if "---" in line:
                continue
            cells = [_inline_html(cell.strip()) for cell in line.strip("|").split("|")]
            tag = "th" if not in_table else "td"
            if not in_table:
                rendered.append("<table><tbody>")
                in_table = True
            rendered.append("<tr>" + "".join(f"<{tag}>{cell}</{tag}>" for cell in cells) + "</tr>")
        else:
            rendered.append(f"<p>{_inline_html(line)}</p>")
    if in_list:
        rendered.append("</ul>")
    if in_ordered_list:
        rendered.append("</ol>")
    if in_table:
        rendered.append("</tbody></table>")
    return rendered


def _inline_html(text: str) -> str:
    escaped = html.escape(text)
    parts = escaped.split("`")
    for index, part in enumerate(parts):
        if index % 2:
            parts[index] = f"<code>{part}</code>"
        else:
            parts[index] = _emphasis_html(part)
    return "".join(parts)


def _emphasis_html(text: str) -> str:
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    return re.sub(r"(?<!\w)_([^_\n]+)_(?!\w)", r"<em>\1</em>", text)


def _ordered_list_item(line: str) -> str | None:
    match = re.match(r"\d+\.\s+(.+)", line)
    if match is None:
        return None
    return match.group(1)


def _markdown_to_pdf(markdown: str) -> bytes:
    try:
        canvas_module = import_module("reportlab.pdfgen.canvas")
        pagesizes_module = import_module("reportlab.lib.pagesizes")
    except ImportError as exc:
        msg = "PDF export requires the optional dependency: pip install 'trainlens[pdf]'."
        raise RuntimeError(msg) from exc

    canvas_class: Any = canvas_module.Canvas
    letter: tuple[float, float] = pagesizes_module.letter
    buffer_module = import_module("io")
    buffer: Any = buffer_module.BytesIO()
    pdf: Any = canvas_class(buffer, pagesize=letter)
    _, height = letter
    x = 54
    y = height - 54
    pdf.setTitle("TrainLens Report")
    pdf.setFont("Helvetica", 10)
    for raw_line in markdown.splitlines():
        line = raw_line.replace("`", "")
        wrapped = textwrap.wrap(line, width=92) or [""]
        for part in wrapped:
            if y < 54:
                pdf.showPage()
                pdf.setFont("Helvetica", 10)
                y = height - 54
            pdf.drawString(x, y, part[:110])
            y -= 14
    pdf.save()
    return bytes(buffer.getvalue())


def _format_from_suffix(path: Path) -> ReportFormat:
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        return "markdown"
    if suffix == ".html":
        return "html"
    if suffix == ".json":
        return "json"
    if suffix == ".pdf":
        return "pdf"
    msg = "Could not infer report format from file extension."
    raise ValueError(msg)


def _assert_never(value: object) -> Never:
    msg = f"Unsupported report format: {value!r}"
    raise ValueError(msg)
