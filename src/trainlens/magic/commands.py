"""IPython magic commands."""

# mypy: disable-error-code="misc,no-untyped-call"

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Any, cast

from IPython.core.magic import Magics, line_magic, magics_class
from IPython.display import Markdown, display

from trainlens.llm.context import build_llm_notebook_context
from trainlens.llm.enhancer import explain_with_llm
from trainlens.pipeline import explain_namespace
from trainlens.renderers.markdown import MarkdownRenderer
from trainlens.storage.memory import InMemoryRunStore


@dataclass(frozen=True)
class _ExplainArguments:
    name: str | None = None
    no_llm: bool = False
    dry_run: bool = False


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
        context = build_llm_notebook_context(shell.user_ns)
        if args.dry_run:
            display(Markdown(context.markdown))
            return

        result = explain_namespace(shell.user_ns)
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
    name: str | None = None
    no_llm = False
    dry_run = False
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--name":
            index += 1
            if index >= len(tokens):
                raise ValueError("--name requires a run label")
            if name is not None:
                raise ValueError("--name can only be provided once")
            name = tokens[index]
        elif token.startswith("--name="):
            if name is not None:
                raise ValueError("--name can only be provided once")
            name = token.partition("=")[2]
            if not name:
                raise ValueError("--name requires a run label")
        elif token == "--no-llm":
            no_llm = True
        elif token == "--dry-run":
            dry_run = True
        else:
            raise ValueError(
                f"Unknown argument {token!r}. Usage: %explain_training "
                "[--name NAME] [--no-llm] [--dry-run]"
            )
        index += 1
    if dry_run and (name is not None or no_llm):
        raise ValueError("--dry-run previews context only and cannot be combined with --name or --no-llm")
    return _ExplainArguments(name=name, no_llm=no_llm, dry_run=dry_run)


def _parse_suggest_arguments(line: str) -> bool:
    tokens = shlex.split(line)
    if not tokens:
        return False
    if tokens == ["--dry-run"]:
        return True
    raise ValueError("Usage: %suggest_improvements [--dry-run]")
