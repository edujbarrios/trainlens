# Changelog

All notable changes to TrainLens will be documented here.

## 0.2.0 - 2026-07-06

- add scientific paper-style LLM report mode with LLM provenance
- add improvement-ideas report mode for follow-up experiments
- expose `build_paper_report` and `build_improvement_ideas` public helpers
- add `%suggest_improvements` notebook magic
- harden secret redaction for URL credentials and query parameters
- validate OpenAI-compatible provider responses with clearer errors
- handle dynamic or symbolic notebook shape values during introspection
- add public-package smoke notebook under ignored local `notebooks/`

## 0.1.0 - 2026-07-02

- initial notebook introspection engine
- training metric heuristics
- IPython magic commands
- optional OpenAI-compatible LLM explanation provider
- Markdown and Rich renderers
- in-memory run comparison
- notebook-only Markdown reporting with no GUI or image dashboard surface
- foundation-model fine-tuning profile detection for LLMs, CLIP, ViTs, projectors, and VLMs
- contrastive, adapter-rank, loss-plateau, and projector-alignment recommendations
- parameterized Jinja2 prompt templates for OpenAI-compatible ML/DL result explanations
- sensitive data redaction for prompt and notebook snapshot safety
