"""Backward-compatible wrapper for the OpenAI-compatible TrainLens helper."""

from __future__ import annotations

from trainlens_openai_compatible import main

if __name__ == "__main__":
    raise SystemExit(main())
