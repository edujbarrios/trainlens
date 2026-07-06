import json

import pytest

from trainlens.llm.config import LLMConfig
from trainlens.llm.openai_compatible import OpenAICompatibleProvider


class FakeResponse:
    def __init__(self, payload: str) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload.encode("utf-8")


def _provider() -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        LLMConfig(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="test-model",
        )
    )


def test_openai_provider_returns_message_content(monkeypatch):
    payload = json.dumps({"choices": [{"message": {"content": "## TrainLens Report"}}]})

    monkeypatch.setattr(
        "trainlens.llm.openai_compatible.request.urlopen",
        lambda *_args, **_kwargs: FakeResponse(payload),
    )

    assert _provider().explain("local evidence") == "## TrainLens Report"


def test_openai_provider_rejects_invalid_json(monkeypatch):
    monkeypatch.setattr(
        "trainlens.llm.openai_compatible.request.urlopen",
        lambda *_args, **_kwargs: FakeResponse("not-json"),
    )

    with pytest.raises(ValueError, match="invalid JSON"):
        _provider().explain("local evidence")


def test_openai_provider_rejects_missing_choices(monkeypatch):
    monkeypatch.setattr(
        "trainlens.llm.openai_compatible.request.urlopen",
        lambda *_args, **_kwargs: FakeResponse(json.dumps({"choices": []})),
    )

    with pytest.raises(ValueError, match="did not include any choices"):
        _provider().explain("local evidence")
