"""Provider retry and typed error regression tests."""

from __future__ import annotations

import json
from urllib.error import HTTPError

import pytest

from trainlens import OpenAICompatibleProvider
from trainlens.llm.openai_compatible import ProviderHTTPError


class Response:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self, _size=-1):
        return json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()


def test_retry_recovers_from_two_service_unavailable_responses(monkeypatch):
    provider = OpenAICompatibleProvider.from_values(
        base_url="https://api.example.com/v1",
        model="model",
        max_retries=2,
        retry_delay_seconds=0,
    )
    calls = 0

    def maybe_fail(req, **_kwargs):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise HTTPError(req.full_url, 503, "Temporary outage", {}, None)
        return Response()

    monkeypatch.setattr("trainlens.llm.openai_compatible.request.urlopen", maybe_fail)
    assert provider.explain("evidence") == "ok"
    assert calls == 3


def test_authentication_errors_are_not_retried(monkeypatch):
    provider = OpenAICompatibleProvider.from_values(
        base_url="https://api.example.com/v1", model="model", max_retries=3
    )
    calls = 0

    def fail(req, **_kwargs):
        nonlocal calls
        calls += 1
        raise HTTPError(req.full_url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr("trainlens.llm.openai_compatible.request.urlopen", fail)
    with pytest.raises(ProviderHTTPError, match="401"):
        provider.explain("evidence")
    assert calls == 1
