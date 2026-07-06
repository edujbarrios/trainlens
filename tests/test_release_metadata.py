from __future__ import annotations

import tomllib
from pathlib import Path

import trainlens


def test_release_versions_are_consistent() -> None:
    root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project_version = pyproject["project"]["version"]
    citation = (root / "CITATION.cff").read_text(encoding="utf-8")

    assert trainlens.__version__ == project_version
    assert f'version: "{project_version}"' in citation
