from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


class FakeResponse:
    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(
            {"choices": [{"message": {"content": "## Improved Report"}}]}
        ).encode("utf-8")


def _load_tool() -> ModuleType:
    root = Path(__file__).resolve().parents[1]
    path = root / "tools" / "trainlens_openai_compatible.py"
    spec = importlib.util.spec_from_file_location("trainlens_openai_compatible_tool", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_openai_compatible_tool_requires_complete_llm_config(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    report = tmp_path / "report.md"
    report.write_text("## Report", encoding="utf-8")
    module = _load_tool()

    monkeypatch.delenv("TRAINLENS_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("TRAINLENS_LLM_API_KEY", raising=False)
    monkeypatch.delenv("TRAINLENS_LLM_MODEL", raising=False)
    monkeypatch.setattr(sys, "argv", ["trainlens_openai_compatible.py", str(report)])

    assert module.main() == 2
    assert "TRAINLENS_LLM_BASE_URL" in capsys.readouterr().err


def test_openai_compatible_tool_sends_configured_model(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    report = tmp_path / "report.md"
    report.write_text("## Report", encoding="utf-8")
    module = _load_tool()
    captured: dict[str, Any] = {}

    def fake_urlopen(req: Any, **_kwargs: object) -> FakeResponse:
        captured["url"] = req.full_url
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setenv("TRAINLENS_LLM_BASE_URL", "https://api.example.com/v1/")
    monkeypatch.setenv("TRAINLENS_LLM_API_KEY", "test-key")
    monkeypatch.setenv("TRAINLENS_LLM_MODEL", "test-model")
    monkeypatch.setattr(sys, "argv", ["trainlens_openai_compatible.py", str(report)])
    monkeypatch.setattr(module.request, "urlopen", fake_urlopen)

    assert module.main() == 0
    assert captured["url"] == "https://api.example.com/v1/chat/completions"
    assert captured["payload"]["model"] == "test-model"
    assert "## Improved Report" in capsys.readouterr().out
