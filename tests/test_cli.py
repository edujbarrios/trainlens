from __future__ import annotations

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
