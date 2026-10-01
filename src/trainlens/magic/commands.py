"""IPython magic commands."""

# mypy: disable-error-code="misc,no-untyped-call"

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Any, cast

from IPython.core.magic import Magics, line_magic, magics_class
from IPython.display import Markdown, display

from trainlens.analysis_config import AnalysisConfig
from trainlens.llm.context import build_llm_notebook_context, build_llm_notebook_context_from_snapshot
from trainlens.llm.enhancer import explain_with_llm
from trainlens.pipeline import analyze_snapshot, snapshot_namespace
from trainlens.renderers.markdown import MarkdownRenderer
from trainlens.storage.memory import InMemoryRunStore


@dataclass(frozen=True)
class _ExplainArguments:
    name: str | None = None
    model: str | None = None
    trainer: str | None = None
    no_llm: bool = False
    dry_run: bool = False
    strict: bool = False


@magics_class
class TrainLensMagics(Magics):
    """Notebook commands for training explanations."""

    def __init__(self, shell: Any = None) -> None:
        super().__init__(shell)
        self.store = InMemoryRunStore()

    @line_magic
    def explain_training(self, line: str = "") -> None:
        args = _parse_explain_arguments(line)
        shell = cast(Any, self.shell)
        snapshot = snapshot_namespace(shell.user_ns)
        config = AnalysisConfig(model=args.model, trainer=args.trainer, strict=args.strict)
        if args.dry_run:
            context = build_llm_notebook_context_from_snapshot(snapshot, analysis_config=config)
            display(Markdown(context.markdown))
            return

        result = analyze_snapshot(snapshot, config=config)
        context = build_llm_notebook_context_from_snapshot(snapshot, analysis_config=config)
        self.store.capture(result, name=args.name)
        if args.no_llm:
            display(Markdown(MarkdownRenderer().render(result)))
            return

        try:
            markdown = explain_with_llm(
                context.markdown,
                mode="paper_report",
                require_provider=True,
            )
        except RuntimeError as exc:
            local = MarkdownRenderer().render(result).rstrip()
            warning = (
                "> **TrainLens:** the run was captured locally, but the LLM explanation "
                f"could not be generated: {exc}\n\n"
            )
            display(Markdown(warning + local + "\n"))
            return
        display(Markdown(markdown))

    @line_magic
    def suggest_improvements(self, line: str = "") -> None:
        dry_run = _parse_suggest_arguments(line)
        shell = cast(Any, self.shell)
        context = build_llm_notebook_context(shell.user_ns)
        if dry_run:
            display(Markdown(context.markdown))
            return
        markdown = explain_with_llm(
            context.markdown,
            mode="improvement_ideas",
            require_provider=True,
        )
        display(Markdown(markdown))

    @line_magic
    def compare_runs(self, line: str = "") -> None:
        selectors = shlex.split(line)
        if not selectors:
            markdown = self.store.render_comparison()
        elif len(selectors) == 2:
            markdown = self.store.render_comparison(selectors[0], selectors[1])
        else:
            raise ValueError(
                "Usage: %compare_runs [BASELINE EXPERIMENT]. "
                "Selectors can be run names or one-based indices."
            )
        display(Markdown(markdown))


def _parse_explain_arguments(line: str) -> _ExplainArguments:
    tokens = shlex.split(line)
    values: dict[str, Any] = {
        "name": None,
        "model": None,
        "trainer": None,
        "no_llm": False,
        "dry_run": False,
        "strict": False,
    }
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in {"--name", "--model", "--trainer"}:
            key = token[2:].replace("-", "_")
            index += 1
            if index >= len(tokens):
                raise ValueError(f"{token} requires a value")
            if values[key] is not None:
                raise ValueError(f"{token} can only be provided once")
            values[key] = tokens[index]
        elif any(token.startswith(f"{prefix}=") for prefix in ("--name", "--model", "--trainer")):
            prefix, _, value = token.partition("=")
            key = prefix[2:].replace("-", "_")
            if values[key] is not None:
                raise ValueError(f"{prefix} can only be provided once")
            if not value:
                raise ValueError(f"{prefix} requires a value")
            values[key] = value
        elif token == "--no-llm":
            values["no_llm"] = True
        elif token == "--dry-run":
            values["dry_run"] = True
        elif token == "--strict":
            values["strict"] = True
        else:
            raise ValueError(
                f"Unknown argument {token!r}. Usage: %explain_training "
                "[--name NAME] [--model MODEL] [--trainer TRAINER] "
                "[--strict] [--no-llm] [--dry-run]"
            )
        index += 1
    if values["dry_run"] and (values["name"] is not None or values["no_llm"]):
        raise ValueError("--dry-run previews context only and cannot be combined with --name or --no-llm")
    return _ExplainArguments(**values)


def _parse_suggest_arguments(line: str) -> bool:
    tokens = shlex.split(line)
    if not tokens:
        return False
    if tokens == ["--dry-run"]:
        return True
    raise ValueError("Usage: %suggest_improvements [--dry-run]")
