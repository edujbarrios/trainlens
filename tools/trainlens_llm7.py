"""Standalone llm7.io helper for TrainLens reports.

This script intentionally uses only the Python standard library so users can run
it without installing TrainLens as a package.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib import request


def main() -> int:
    parser = argparse.ArgumentParser(description="Enhance a TrainLens report with llm7.io.")
    parser.add_argument(
        "report",
        nargs="?",
        help="Path to a Markdown report. Reads stdin when omitted.",
    )
    args = parser.parse_args()

    report = Path(args.report).read_text(encoding="utf-8") if args.report else sys.stdin.read()
    base_url = os.getenv("TRAINLENS_LLM_BASE_URL", "https://api.llm7.io/v1").rstrip("/")
    api_key = os.getenv("TRAINLENS_LLM_API_KEY")
    model = os.getenv("TRAINLENS_LLM_MODEL", "auto")
    if not api_key:
        print("TRAINLENS_LLM_API_KEY is required.", file=sys.stderr)
        return 2

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Improve this ML training report without inventing facts.",
            },
            {"role": "user", "content": report},
        ],
    }
    req = request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=30) as response:  # noqa: S310
        data = json.loads(response.read().decode("utf-8"))
    print(data["choices"][0]["message"]["content"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
