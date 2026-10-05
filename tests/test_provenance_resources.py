from __future__ import annotations

from trainlens import (
    TrainingRun,
    attach_resource_profile,
    capture_provenance,
    fingerprint_data,
    profile_resources,
)
from trainlens.models.metric import MetricSeries


def test_fingerprint_data_is_deterministic_for_json_and_files(tmp_path) -> None:
    left = fingerprint_data({"b": [2, 3], "a": 1})
    right = fingerprint_data({"a": 1, "b": [2, 3]})
    assert left == right
    assert left.startswith("sha256:")

    path = tmp_path / "dataset.txt"
    path.write_bytes(b"trainlens")
    assert fingerprint_data(path) == fingerprint_data(b"trainlens")


def test_capture_provenance_flattens_safe_metadata(tmp_path) -> None:
    provenance = capture_provenance(
        seed=42,
        dataset=[{"x": 1, "y": 0}, {"x": 2, "y": 1}],
        packages=("trainlens", "definitely-not-installed-trainlens-test"),
        cwd=tmp_path,
    )

    metadata = provenance.as_metadata()
    assert metadata["provenance.seed"] == 42
    assert str(metadata["provenance.dataset_fingerprint"]).startswith("sha256:")
    assert metadata["provenance.python_version"]
    assert "provenance.package.definitely-not-installed-trainlens-test" not in metadata


def test_resource_profile_exposes_objective_metrics_and_attaches_to_run() -> None:
    profile = profile_resources(
        duration_seconds=20,
        items_processed=100,
        tokens_processed=1000,
        peak_memory_mb=2048,
        gpu_memory_mb=1024,
        estimated_cost_usd=0.25,
    )

    assert profile.throughput_per_second == 5
    assert profile.token_throughput_per_second == 50
    assert profile.metrics()["estimated_cost_usd"] == 0.25

    run = TrainingRun(
        run_id="run",
        metrics=(MetricSeries("validation_loss", (0.5,)),),
    )
    enriched = attach_resource_profile(run, profile)
    metrics = {series.name: series.last for series in enriched.metrics}
    assert metrics["validation_loss"] == 0.5
    assert metrics["duration_seconds"] == 20
    assert metrics["throughput_per_second"] == 5
    assert enriched.run_id == run.run_id
