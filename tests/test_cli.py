from __future__ import annotations

import json

from trainlens.cli import main
from trainlens.models.metric import MetricSeries
from trainlens.models.run import TrainingRun
from trainlens.runs import save_run


def _write_run(tmp_path, name: str, loss: float, f1: float):
    path = tmp_path / f"{name}.json"
    save_run(
        TrainingRun(
            run_id=name,
            metrics=(
                MetricSeries("validation_loss", (loss,)),
                MetricSeries("f1", (f1,)),
            ),
        ),
        path,
    )
    return path


def test_cli_compare_renders_portable_runs(tmp_path, capsys) -> None:
    baseline = _write_run(tmp_path, "baseline", 0.5, 0.80)
    candidate = _write_run(tmp_path, "candidate", 0.4, 0.85)

    assert main(["compare", str(baseline), str(candidate)]) == 0
    output = capsys.readouterr().out
    assert "TrainLens Run Comparison" in output
    assert "validation_loss" in output


def test_cli_check_fails_on_material_regression(tmp_path, capsys) -> None:
    baseline = _write_run(tmp_path, "baseline", 0.5, 0.80)
    candidate = _write_run(tmp_path, "candidate", 0.65, 0.80)

    assert main(["check", str(candidate), "--against", str(baseline)]) == 1
    assert "material regression: validation_loss" in capsys.readouterr().out


def test_cli_check_supports_metric_gates(tmp_path, capsys) -> None:
    candidate = _write_run(tmp_path, "candidate", 0.4, 0.86)

    assert main(["check", str(candidate), "--require", "f1>=0.85"]) == 0
    assert "PASS" in capsys.readouterr().out

    assert main(["check", str(candidate), "--require", "validation_loss<=0.3"]) == 1
    assert "gate failed" in capsys.readouterr().out

    assert main(["check", str(candidate), "--require", "latency_ms<=10"]) == 1
    assert "missing required metric" in capsys.readouterr().out


def test_cli_split_aware_gates_do_not_overwrite_metric_names(tmp_path, capsys) -> None:
    path = tmp_path / "split.json"
    save_run(
        TrainingRun(
            run_id="split",
            metrics=(
                MetricSeries("loss", (0.1,), split="train"),
                MetricSeries("loss", (0.4,), split="validation"),
            ),
        ),
        path,
    )
    assert main(["check", str(path), "--require", "validation_loss<=0.5"]) == 0
    assert main(["check", str(path), "--require", "train_loss<=0.05"]) == 1
    assert "gate failed: train_loss" in capsys.readouterr().out


def test_cli_rejects_missing_baseline_metrics_unless_explicitly_allowed(
    tmp_path, capsys
) -> None:
    baseline = _write_run(tmp_path, "baseline", 0.5, 0.8)
    candidate = tmp_path / "missing.json"
    save_run(TrainingRun(run_id="missing"), candidate)

    assert main(["check", str(candidate), "--against", str(baseline)]) == 1
    assert "removed baseline metric" in capsys.readouterr().out
    assert main(
        ["check", str(candidate), "--against", str(baseline), "--allow-removed"]
    ) == 0


def test_check_json_output_and_exit_codes(tmp_path, capsys) -> None:
    candidate = _write_run(tmp_path, "candidate", 0.4, 0.85)
    args = ["check", str(candidate), "--format", "json", "--require", "f1>=0.9"]
    assert main(args) == 1
    output = json.loads(capsys.readouterr().out)
    assert output["passed"] is False
    assert len(output["failures"]) == 1
    assert "gate failed" in output["failures"][0]

    destination = tmp_path / "result.json"
    assert main(["check", str(candidate), "--format", "json", "--output", str(destination)]) == 0
    assert json.loads(destination.read_text()) == {"passed": True, "failures": []}
