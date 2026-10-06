"""Local project storage for portable TrainLens runs."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

from trainlens.analysis_config import AnalysisConfig
from trainlens.models.run import RunValue, TrainingRun
from trainlens.pipeline import analyze
from trainlens.runs import load_run, save_run, training_run_from_analysis

_ProjectMeta: TypeAlias = tuple[str | None, tuple[str, ...]]


@dataclass(frozen=True)
class ProjectEntry:
    """One stored run together with project-level labels."""

    run: TrainingRun
    name: str | None = None
    tags: tuple[str, ...] = ()


class Project:
    """A lightweight local registry backed by TrainLens JSON run files."""

    def __init__(self, root: str | Path = ".trainlens") -> None:
        self.root = Path(root)
        self._runs_dir = self.root / "runs"
        self._index_path = self.root / "index.json"

    def add(
        self,
        run: TrainingRun,
        *,
        name: str | None = None,
        tags: Sequence[str] = (),
        overwrite: bool = False,
    ) -> ProjectEntry:
        """Persist a run and optional project labels."""

        clean_name = _validate_name(name)
        clean_tags = _validate_tags(tags)
        with self._write_lock():
            index = self._read_index()
            if run.run_id in index and not overwrite:
                raise FileExistsError(f"run {run.run_id!r} already exists in this project")

            self._runs_dir.mkdir(parents=True, exist_ok=True)
            path = self._run_path(run.run_id)
            temporary = path.with_name(f".{path.name}.tmp")
            save_run(run, temporary)
            temporary.replace(path)
            index[run.run_id] = (clean_name, clean_tags)
            self._write_index(index)
        return ProjectEntry(run=run, name=clean_name, tags=clean_tags)

    def capture(
        self,
        namespace: Mapping[str, Any],
        *,
        name: str | None = None,
        tags: Sequence[str] = (),
        config: AnalysisConfig | None = None,
        run_id: str | None = None,
        parameters: Mapping[str, RunValue] | None = None,
        metadata: Mapping[str, RunValue] | None = None,
        notes: tuple[str, ...] = (),
        overwrite: bool = False,
    ) -> ProjectEntry:
        """Analyze a namespace, convert it to a portable run, and persist it."""

        result = analyze(namespace, config=config)
        run = training_run_from_analysis(
            result,
            run_id=run_id,
            parameters=parameters,
            metadata=metadata,
            notes=notes,
        )
        return self.add(run, name=name, tags=tags, overwrite=overwrite)

    def get(self, run_id: str) -> ProjectEntry:
        """Load one stored run by id."""

        index = self._read_index()
        metadata = index.get(run_id)
        if metadata is None:
            raise KeyError(run_id)
        path = self._run_path(run_id)
        if not path.is_file():
            raise FileNotFoundError(f"project index references missing run file: {path}")
        name, tags = metadata
        return ProjectEntry(run=load_run(path), name=name, tags=tags)

    def entries(self) -> tuple[ProjectEntry, ...]:
        """Return all project entries ordered by creation time and run id."""

        entries = tuple(self.get(run_id) for run_id in self._read_index())
        return tuple(sorted(entries, key=lambda item: (item.run.created_at, item.run.run_id)))

    def runs(self) -> tuple[TrainingRun, ...]:
        """Return all stored training runs."""

        return tuple(entry.run for entry in self.entries())

    def find(
        self,
        *,
        name: str | None = None,
        tag: str | None = None,
    ) -> tuple[ProjectEntry, ...]:
        """Filter stored runs by exact project name and/or tag."""

        return tuple(
            entry
            for entry in self.entries()
            if (name is None or entry.name == name) and (tag is None or tag in entry.tags)
        )

    def remove(self, run_id: str) -> None:
        """Remove one run and its project metadata."""

        with self._write_lock():
            index = self._read_index()
            if run_id not in index:
                raise KeyError(run_id)
            path = self._run_path(run_id)
            if path.exists():
                path.unlink()
            del index[run_id]
            self._write_index(index)

    def __len__(self) -> int:
        return len(self._read_index())

    @contextmanager
    def _write_lock(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        lock_path = self.root / ".project.lock"
        deadline = time.monotonic() + 10.0
        descriptor: int | None = None
        while descriptor is None:
            try:
                descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(descriptor, f"{os.getpid()}\n".encode())
            except FileExistsError:
                try:
                    stale = time.time() - lock_path.stat().st_mtime > 60.0
                except FileNotFoundError:
                    continue
                if stale:
                    try:
                        lock_path.unlink()
                    except FileNotFoundError:
                        pass
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out waiting for TrainLens project lock: {lock_path}")
                time.sleep(0.05)
        try:
            yield
        finally:
            os.close(descriptor)
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass

    def _run_path(self, run_id: str) -> Path:
        if not run_id or Path(run_id).name != run_id:
            raise ValueError("run_id must be a non-empty filename-safe identifier")
        return self._runs_dir / f"{run_id}.json"

    def _read_index(self) -> dict[str, _ProjectMeta]:
        if not self._index_path.exists():
            return {}
        raw: object = json.loads(self._index_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("schema_version") != 1:
            raise ValueError("unsupported or invalid TrainLens project index")
        runs_raw = raw.get("runs")
        if not isinstance(runs_raw, dict):
            raise ValueError("TrainLens project index runs must be an object")

        output: dict[str, _ProjectMeta] = {}
        for run_id, value in runs_raw.items():
            if not isinstance(run_id, str) or not isinstance(value, dict):
                raise ValueError("TrainLens project index contains an invalid run entry")
            name = value.get("name")
            tags = value.get("tags", [])
            if name is not None and not isinstance(name, str):
                raise ValueError("TrainLens project run name must be a string or null")
            if not isinstance(tags, list) or not all(isinstance(item, str) for item in tags):
                raise ValueError("TrainLens project run tags must be strings")
            output[run_id] = (name, tuple(tags))
        return output

    def _write_index(self, index: Mapping[str, _ProjectMeta]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "runs": {
                run_id: {"name": name, "tags": list(tags)}
                for run_id, (name, tags) in sorted(index.items())
            },
        }
        temporary = self._index_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self._index_path)


def _validate_name(name: str | None) -> str | None:
    if name is None:
        return None
    if not isinstance(name, str) or not name.strip():
        raise ValueError("project run name must be a non-empty string")
    return name.strip()


def _validate_tags(tags: Sequence[str]) -> tuple[str, ...]:
    clean: list[str] = []
    for tag in tags:
        if not isinstance(tag, str) or not tag.strip():
            raise ValueError("project run tags must be non-empty strings")
        normalized = tag.strip()
        if normalized not in clean:
            clean.append(normalized)
    return tuple(clean)
