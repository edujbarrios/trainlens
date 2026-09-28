import pytest

from trainlens.llm.config import LLMConfig
from trainlens.llm.openai_compatible import (
    _MAX_RESPONSE_BYTES,
    OpenAICompatibleProvider,
)


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.requested_size: int | None = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        self.requested_size = size
        if self.body:
            return self.body[:size]
        return b"x" * size


def _provider() -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        LLMConfig(base_url="https://example.invalid/v1", api_key="secret", model="demo")
    )


def test_provider_bounds_response_reads(monkeypatch) -> None:
    response = FakeResponse(b"")
    monkeypatch.setattr(
        "trainlens.llm.openai_compatible.request.urlopen",
        lambda *_args, **_kwargs: response,
    )

    with pytest.raises(ValueError, match="exceeded the maximum supported size"):
        _provider().explain("# report")

    assert response.requested_size == _MAX_RESPONSE_BYTES + 1


def test_provider_rejects_invalid_utf8(monkeypatch) -> None:
    response = FakeResponse(b"\xff")
    monkeypatch.setattr(
        "trainlens.llm.openai_compatible.request.urlopen",
        lambda *_args, **_kwargs: response,
    )

    with pytest.raises(ValueError, match="invalid UTF-8"):
        _provider().explain("# report")
