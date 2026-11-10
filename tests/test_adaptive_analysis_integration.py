"""Opt-in wiring for adaptive diagnostics."""

from trainlens import AnalysisConfig, analyze, diagnose_trend
from trainlens.models.metric import MetricSeries


def test_adaptive_diagnostics_are_opt_in() -> None:
    state = {"history": {
        "train_loss": [1.0, 0.9, 0.8, 0.7, 0.6],
        "val_loss": [0.7, 0.8, 0.9, 1.0, 1.1],
    }}
    default = analyze(state)
    enabled = analyze(state, config=AnalysisConfig(adaptive_diagnostics=True))
    assert not any(signal.title.startswith("Adaptive trend:") for signal in default.signals)
    assert any(signal.title == "Adaptive trend: worsening" for signal in enabled.signals)


def test_adaptive_api_is_public() -> None:
    result = diagnose_trend(MetricSeries("validation_loss", (1.0, 0.8, 0.6, 0.4, 0.2)))
    assert result is not None
    assert result.state == "improving"
