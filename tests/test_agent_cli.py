from __future__ import annotations

import json

from trainlens.cli import main
from trainlens.models.metric import MetricSeries
from trainlens.models.run import TrainingRun
from trainlens.runs import save_run


def _write_run(tmp_path):
    path = tmp_path / "candidate.json"
    save_run(
        TrainingRun(
            run_id="candidate",
            metrics=(
                MetricSeries("train_loss", (1.0, 0.7, 0.4)),
                MetricSeries("validation_loss", (1.0, 0.65, 0.68)),
            ),
            parameters={"learning_rate": 2e-5},
        ),
        path,
    )
    return path


def test_cli_agent_context_and_verification_round_trip(tmp_path, capsys) -> None:
    run_path = _write_run(tmp_path)
    context_path = tmp_path / "agent-context.json"

    assert main(["agent-context", str(run_path), "--output", str(context_path)]) == 0
    context_payload = json.loads(context_path.read_text(encoding="utf-8"))
    assert context_payload["mode"] == "agent"
    assert context_payload["evidence"]

    evidence_id = context_payload["evidence"][0]["evidence_id"]
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "recommendations": [
                    {
                        "action": "run one controlled follow-up",
                        "rationale": "test the highest-value hypothesis",
                        "evidence_ids": [evidence_id],
                        "confidence": 0.7,
                        "success_criterion": "validation evidence improves",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert main(["verify-agent-plan", str(plan_path), "--context", str(context_path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["fully_supported"] is True
