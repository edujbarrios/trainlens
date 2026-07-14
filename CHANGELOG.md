# Changelog

All notable changes to TrainLens will be documented here.

## 0.7.0 - 2026-07-14

- ignore non-finite metrics in run comparisons to prevent invalid deltas and JSON values
- match complete metric-name tokens when inferring whether higher or lower values are better
- compact long metric histories while preserving endpoints, extrema, counts, and ordered samples
- avoid repeating metric-container literals in LLM notebook context
- allow LLM report callers to tune curve detail with `max_metric_points`

## 0.6.0 - 2026-07-10

- add public `compare_runs(baseline, experiment)` API for comparing training metrics across runs
- classify metric changes as improvements, regressions, unchanged, new, removed, or unknown
- render run comparisons as Markdown, HTML, and JSON through the existing export helpers
- use structured run comparison output in the notebook run store when at least two runs are captured

## 0.5.0 - 2026-07-10

- add lightweight framework adapters for Keras History objects, Hugging Face Trainer logs, and PyTorch Lightning Trainer metrics
- merge adapted framework metrics into the existing analysis pipeline without requiring TensorFlow, Transformers, PyTorch, or Lightning as dependencies
- report adapted framework evidence in notebook summaries and improve metric discovery for adapter-generated metric mappings

## 0.4.0 - 2026-07-09

- add `render_report` and `write_report` helpers for Markdown, HTML, JSON, and optional PDF export
- expose report export helpers from the public `trainlens` API
- document report export usage and the optional `trainlens[pdf]` extra

## 0.3.0 - 2026-07-06

- simplify README setup and local Ollama configuration guidance
- remove committed example notebook history after accidental credential exposure
- remove the `examples/` tree from repository history
- bound nested notebook literal redaction to avoid oversized LLM contexts
- normalize slash and dash metric names such as `train/accuracy` and `eval-accuracy`
- align README, `.env.example`, and standalone helper LLM configuration
- avoid contrastive-loss warnings for generic validation-loss regressions
- refresh package and citation metadata for the 0.3.0 release

## 0.2.0 - 2026-07-06

- add scientific paper-style LLM report mode with LLM provenance
- add improvement-ideas report mode for follow-up experiments
- expose `build_paper_report` and `build_improvement_ideas` public helpers
- add `%suggest_improvements` notebook magic
- harden secret redaction for URL credentials and query parameters
- validate OpenAI-compatible provider responses with clearer errors
- handle dynamic or symbolic notebook shape values during introspection

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
