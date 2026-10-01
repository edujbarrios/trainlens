import json
from typing import Any

from trainlens import LLMConfig, OpenAICompatibleProvider, build_paper_report


class FakeResponse:
    def __init__(self, payload: str) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        encoded = self.payload.encode("utf-8")
        return encoded if size < 0 else encoded[:size]


def test_llm_config_allows_missing_api_key_for_local_endpoints(monkeypatch):
    monkeypatch.setenv("TRAINLENS_LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("TRAINLENS_LLM_MODEL", "local-model")
    monkeypatch.delenv("TRAINLENS_LLM_API_KEY", raising=False)

    config = LLMConfig.from_env()

    assert config is not None
    assert config.api_key is None
    assert config.base_url == "http://localhost:11434/v1"


def test_openai_compatible_provider_omits_authorization_without_key(monkeypatch):
    captured: dict[str, Any] = {}
    payload = json.dumps({"choices": [{"message": {"content": "report"}}]})
    provider = OpenAICompatibleProvider(
        LLMConfig(
            base_url="http://localhost:11434/v1",
            model="local-model",
        )
    )

    def fake_urlopen(req: Any, **_kwargs: object) -> FakeResponse:
        captured["authorization"] = req.get_header("Authorization")
        return FakeResponse(payload)

    monkeypatch.setattr("trainlens.llm.openai_compatible.request.urlopen", fake_urlopen)

    assert provider.explain("local evidence") == "report"
    assert captured["authorization"] is None


def test_build_paper_report_accepts_injected_provider_without_environment(monkeypatch):
    captured: dict[str, object] = {}

    class CustomProvider:
        def explain(
            self,
            markdown_report: str,
            *,
            mode: str = "paper_report",
            prompt_options: object | None = None,
        ) -> str:
            captured["markdown"] = markdown_report
            captured["mode"] = mode
            captured["prompt_options"] = prompt_options
            return "## Custom provider report"

    monkeypatch.delenv("TRAINLENS_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("TRAINLENS_LLM_API_KEY", raising=False)
    monkeypatch.delenv("TRAINLENS_LLM_MODEL", raising=False)

    report = build_paper_report(
        {"history": {"train_loss": [1.0, 0.6], "eval_loss": [1.1, 0.8]}},
        provider=CustomProvider(),
    )

    assert report.markdown == "## Custom provider report"
    assert captured["mode"] == "paper_report"
    assert "TrainLens Notebook Context" in str(captured["markdown"])
