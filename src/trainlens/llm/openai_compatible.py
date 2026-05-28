"""OpenAI-compatible chat completions provider."""

from __future__ import annotations

import json
from dataclasses import dataclass
from urllib import request

from trainlens.llm.config import LLMConfig
from trainlens.llm.prompts import render_report_enhancement_prompt


@dataclass
class OpenAICompatibleProvider:
    config: LLMConfig

    def enhance(self, markdown_report: str) -> str:
        prompt = render_report_enhancement_prompt(markdown_report)
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": prompt,
                },
                {
                    "role": "user",
                    "content": "Enhance the TrainLens report using the rendered instructions.",
                },
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.config.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=20) as response:  # noqa: S310
            data = json.loads(response.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
        return str(content)
