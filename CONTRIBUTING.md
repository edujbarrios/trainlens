# Contributing to TrainLens

Thanks for considering a contribution. TrainLens is designed to be contributor-friendly from day one.

## Ways to help

- add analyzers for new training patterns
- improve notebook magic UX
- write example notebooks
- expand heuristic coverage
- document edge cases
- add provider adapters

## Development

```bash
python -m venv .venv
pip install -e ".[dev]"
pytest
ruff check .
mypy src/trainlens
```

## Design principles

- local heuristic behavior comes first
- LLMs may enhance language, but must not be required
- analyzers should explain their evidence
- notebook output should be concise and scannable
- integrations must avoid importing heavy ML frameworks unless the user already has them loaded

## Pull requests

Keep pull requests focused. Include tests for new analyzers and mention any notebook UX changes in the PR description.
