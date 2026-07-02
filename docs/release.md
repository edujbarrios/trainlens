# Release Process

TrainLens publishes source distributions and wheels to PyPI from a clean main
branch.

## Preflight

```bash
python -m pip install -e ".[dev]"
ruff check .
mypy src/trainlens
python -m pytest
```

Confirm the release metadata:

- `pyproject.toml` has the intended version.
- `CHANGELOG.md` has a dated entry for the version.
- `README.md` describes the PyPI install path.
- `https://pypi.org/project/trainlens/` is available before the first publish.

## Build

```bash
python -m build
python -m twine check dist/*
```

Inspect the generated files in `dist/` before uploading.

## Publish

Upload to TestPyPI first when changing packaging metadata:

```bash
python -m twine upload --repository testpypi dist/*
```

Publish to PyPI:

```bash
python -m twine upload dist/*
```

After publishing, create and push a matching Git tag, for example:

```bash
git tag v0.1.0
git push origin v0.1.0
```
