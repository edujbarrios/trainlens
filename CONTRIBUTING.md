# Contributing to TrainLens

Thanks for helping improve TrainLens. Small, focused fixes to docs, examples, tests, and analyzers are welcome.

## Development setup

Requires Python 3.11–3.14.

```bash
git clone https://github.com/edujbarrios/trainlens.git
cd trainlens
python -m venv .venv
# Activate .venv using your shell's usual command.
python -m pip install -e ".[dev]"
```

Run the checks relevant to your changes:

```bash
python -m pytest
ruff check .
mypy src/trainlens
```

For documentation-only changes, at minimum check Markdown links and examples; the install/documentation guard can be run with `python -m pytest tests/test_documentation_install.py`. You do **not** need to build or publish a package for a docs-only change.

## Principles

- Keep deterministic analysis useful without an LLM.
- Explain findings with evidence; do not claim causal certainty.
- Keep notebook output and public docs short and readable.
- Avoid importing optional ML frameworks unless the user has them installed.
- Never commit credentials, executed notebook outputs, or private training data.

## Pull requests

Keep each PR focused. Explain what changed, include tests for behavioral changes, and mention any impact on notebook UX. Documentation and repository-maintenance updates do not require a version bump or a release. Release work is handled separately.
