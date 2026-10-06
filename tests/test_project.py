from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pytest

from trainlens import Project, TrainingRun
from trainlens.models.metric import MetricSeries


def _run(run_id: str, value: float = 0.5) -> TrainingRun:
    return TrainingRun(
        run_id=run_id,
        created_at=datetime(2026, 10, 5, 10, 0, tzinfo=UTC),
        metrics=(MetricSeries("validation_loss", (value,)),),
        parameters={"learning_rate": 0.001},
        metadata={"seed": 42},
    )


def test_project_persists_labels_and_filters_runs(tmp_path) -> None:
    project = Project(tmp_path / ".trainlens")
    first = project.add(_run("baseline", 0.5), name="baseline", tags=("lora", "baseline"))
    second = project.add(_run("candidate", 0.4), name="candidate", tags=("lora", "candidate"))

    assert len(project) == 2
    assert project.get("baseline") == first
    assert project.runs() == (first.run, second.run)
    assert project.find(tag="candidate") == (second,)
    assert project.find(name="baseline") == (first,)

    restored = Project(tmp_path / ".trainlens")
    assert restored.entries() == (first, second)

    restored.remove("baseline")
    assert len(restored) == 1
    with pytest.raises(KeyError):
        restored.get("baseline")


def test_project_duplicate_overwrite_and_validation(tmp_path) -> None:
    project = Project(tmp_path)
    project.add(_run("same"), tags=(" tuned ", "tuned"))

    with pytest.raises(FileExistsError):
        project.add(_run("same", 0.3))

    entry = project.add(_run("same", 0.3), name=" winner ", overwrite=True)
    assert entry.name == "winner"
    assert project.get("same").run.metrics[0].last == 0.3

    with pytest.raises(ValueError):
        project.add(_run("../unsafe"))
    with pytest.raises(ValueError):
        project.add(_run("bad-name"), name=" ")
    with pytest.raises(ValueError):
        project.add(_run("bad-tag"), tags=("",))
    with pytest.raises(KeyError):
        project.remove("missing")


def test_project_capture_analyzes_and_saves_namespace(tmp_path) -> None:
    project = Project(tmp_path)

    entry = project.capture(
        {
            "history": {
                "train_loss": [1.0, 0.7, 0.5],
                "eval_loss": [1.1, 0.8, 0.6],
            }
        },
        run_id="captured",
        name="notebook-run",
        tags=("captured",),
        parameters={"learning_rate": 0.001},
    )

    assert entry.run.run_id == "captured"
    assert entry.name == "notebook-run"
    assert entry.run.parameters["learning_rate"] == 0.001
    series = {metric.name: metric for metric in entry.run.metrics}
    assert series["train_loss"].values == (1.0, 0.7, 0.5)
    assert series["validation_loss"].values == (1.1, 0.8, 0.6)
    assert project.get("captured") == entry


def test_project_serializes_concurrent_writers_without_lost_index_entries(tmp_path) -> None:
    project = Project(tmp_path / ".trainlens")

    def add_run(index: int) -> None:
        project.add(_run(f"run-{index}", value=0.5 - index * 0.01))

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(add_run, range(8)))

    assert len(project) == 8
    assert {entry.run.run_id for entry in project.entries()} == {
        f"run-{index}" for index in range(8)
    }
    assert not (project.root / ".project.lock").exists()
