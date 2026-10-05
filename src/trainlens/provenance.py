"""Dependency-free reproducibility provenance for training runs."""

from __future__ import annotations

import hashlib
import json
import platform as platform_module
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from trainlens.models.run import RunValue


@dataclass(frozen=True)
class RunProvenance:
    """Portable environment and source provenance for one experiment."""

    python_version: str
    platform: str
    git_commit: str | None = None
    git_dirty: bool | None = None
    seed: int | None = None
    dataset_fingerprint: str | None = None
    packages: Mapping[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.packages is None:
            object.__setattr__(self, "packages", {})

    def as_metadata(self, *, prefix: str = "provenance") -> dict[str, RunValue]:
        """Flatten provenance into scalar metadata accepted by :class:`TrainingRun`."""

        metadata: dict[str, RunValue] = {
            f"{prefix}.python_version": self.python_version,
            f"{prefix}.platform": self.platform,
        }
        if self.git_commit is not None:
            metadata[f"{prefix}.git_commit"] = self.git_commit
        if self.git_dirty is not None:
            metadata[f"{prefix}.git_dirty"] = self.git_dirty
        if self.seed is not None:
            metadata[f"{prefix}.seed"] = self.seed
        if self.dataset_fingerprint is not None:
            metadata[f"{prefix}.dataset_fingerprint"] = self.dataset_fingerprint
        for name, package_version in sorted(self.packages.items()):
            metadata[f"{prefix}.package.{name}"] = package_version
        return metadata


def capture_provenance(
    *,
    seed: int | None = None,
    dataset: Any | None = None,
    packages: Sequence[str] = (),
    cwd: str | Path | None = None,
) -> RunProvenance:
    """Capture safe local provenance without importing ML frameworks.

    Package versions are read from installed distribution metadata. Missing requested
    packages are omitted. Git information is best-effort and never raises when the working
    directory is not a repository or Git is unavailable.
    """

    if seed is not None and isinstance(seed, bool):
        raise TypeError("seed must be an integer or null")
    if seed is not None and not isinstance(seed, int):
        raise TypeError("seed must be an integer or null")
    commit, dirty = _git_state(cwd)
    return RunProvenance(
        python_version=platform_module.python_version(),
        platform=platform_module.platform(),
        git_commit=commit,
        git_dirty=dirty,
        seed=seed,
        dataset_fingerprint=None if dataset is None else fingerprint_data(dataset),
        packages=_package_versions(packages),
    )


def fingerprint_data(value: Any) -> str:
    """Return a deterministic SHA-256 fingerprint for files or JSON-like data."""

    digest = hashlib.sha256()
    if isinstance(value, Path):
        if not value.is_file():
            raise ValueError(f"dataset path is not a file: {value}")
        with value.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return f"sha256:{digest.hexdigest()}"
    if isinstance(value, (bytes, bytearray, memoryview)):
        digest.update(bytes(value))
        return f"sha256:{digest.hexdigest()}"

    candidate = value
    tolist = getattr(candidate, "tolist", None)
    if callable(tolist):
        candidate = tolist()
    try:
        encoded = json.dumps(
            candidate,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "dataset fingerprinting supports files, bytes, arrays with tolist(), or JSON-like data"
        ) from exc
    digest.update(encoded)
    return f"sha256:{digest.hexdigest()}"


def _package_versions(packages: Sequence[str]) -> dict[str, str]:
    output: dict[str, str] = {}
    for package in packages:
        if not isinstance(package, str) or not package.strip():
            raise ValueError("package names must be non-empty strings")
        normalized = package.strip()
        try:
            output[normalized] = version(normalized)
        except PackageNotFoundError:
            continue
    return output


def _git_state(cwd: str | Path | None) -> tuple[str | None, bool | None]:
    working_directory = None if cwd is None else str(cwd)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=working_directory,
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if commit.returncode != 0:
            return None, None
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=working_directory,
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None, None
    dirty = None if status.returncode != 0 else bool(status.stdout.strip())
    return commit.stdout.strip() or None, dirty
