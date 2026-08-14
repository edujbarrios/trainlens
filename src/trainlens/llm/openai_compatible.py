"""OpenAI-compatible chat completions provider."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib import request

from trainlens.llm.config import LLMConfig
from trainlens.llm.prompts import (
    PromptOptions,
    ReportMode,
    render_ml_results_explanation_prompt,
    render_prompt_with_options,
)


@dataclass
class OpenAICompatibleProvider:
    config: LLMConfig

    def explain(
        self,
        markdown_report: str,
        *,
        mode: ReportMode = "paper_report",
        prompt_options: PromptOptions | None = None,
    ) -> str:
        if prompt_options is None:
            prompt = render_ml_results_explanation_prompt(
                markdown_report, mode=mode, llm_model=self.config.model
            )
        else:
            prompt = render_prompt_with_options(
                markdown_report,
                options=prompt_options,
                mode=mode,
                llm_model=self.config.model,
            )
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": prompt,
                },
                {
                    "role": "user",
                    "content": "Explain the ML/DL results using the provided context.",
                },
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.config.base_url.rstrip('/')}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=self.config.timeout_seconds) as response:  # noqa: S310
            raw_response = response.read().decode("utf-8")
        try:
            data = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            msg = "LLM provider returned invalid JSON."
            raise ValueError(msg) from exc
        return _extract_message_content(data)


def _extract_message_content(data: Any) -> str:
    if not isinstance(data, dict):
        msg = "LLM provider response must be a JSON object."
        raise ValueError(msg)
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        msg = "LLM provider response did not include any choices."
        raise ValueError(msg)
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        msg = "LLM provider choice must be a JSON object."
        raise ValueError(msg)
    message = first_choice.get("message")
    if not isinstance(message, dict):
        msg = "LLM provider choice did not include a message object."
        raise ValueError(msg)
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        msg = "LLM provider message content was empty or invalid."
        raise ValueError(msg)
    return content
