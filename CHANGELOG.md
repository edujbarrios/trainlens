# Changelog

All notable changes to TrainLens will be documented here.

## 0.9.0 - 2026-09-25

- preserve metric history integrity with aligned step metadata, finite-value filtering, fractional epochs, and bounded one-dimensional array-like histories
- centralize metric optimization semantics and natural bounds across run comparison and next-experiment recommendations
- add persistent alert lifecycle handling so stagnation and overfitting alerts re-arm only after recovery
- add optional framework-native callback adapters for Keras 3, Hugging Face Transformers, and Lightning while retaining the dependency-free callback core
- make notebook capture local-first, including offline `--no-llm` runs, provider-failure fallback, stable run names, explicit run selection, and full `AnalysisResult` metadata
- add exact outbound-context preview via `preview_notebook_context()` and `%explain_training --dry-run`, plus aligned metric steps and fractional epochs in bounded LLM context
- render notebook reports natively as Markdown, deduplicate model candidates, preserve extension state across reloads, and add real IPython integration tests
- continuously test and advertise CPython 3.11, 3.12, 3.13, and 3.14 with an explicit `<3.15` upper bound

## 0.8.6 - 2026-09-24

- keep notebook introspection running when unrelated objects expose failing properties or model/framework probes
- minimize outbound LLM context by omitting unrelated literal notebook values by default while allowing explicit sanitized opt-in
- keep notebook-derived evidence out of trusted system instructions and treat it as separate untrusted provider input
- document the new LLM privacy defaults and prompt-injection trust boundary in the README

## 0.8.5 - 2026-09-24

- preserve single-point training accuracy values while keeping change summaries delta-dependent
- redact fine-grained GitHub personal access tokens and secret-shaped mapping keys from notebook context
- avoid false class-imbalance warnings on balanced multiclass datasets by scaling against the balanced class expectation

## 0.8.4 - 2026-09-14

- redact camelCase credential fields in notebook and framework configuration
- include Hugging Face and Lightning adapter metrics in LLM notebook context
- exclude optimizers, schedulers, and data loaders from model candidates
- distinguish adapted metrics from captured framework training parameters
- tolerate broken framework log-history containers without boolean evaluation
- reject success criteria beyond natural metric bounds
- reject backwards monitoring steps and boolean trace steps
- capture independent in-memory snapshots of mutable analysis results

## 0.8.3 - 2026-09-14

- redact secrets from every customizable prompt field before provider submission
- preserve Markdown and HTML table structure for arbitrary metric and trace names
- reject text and mappings as class-label collections
- avoid interpreting non-finite loss values as a valid generalization gap
- avoid boolean evaluation of framework model references and log histories
- validate LLM metric-point budgets and count metric-only results as findings
- ignore boolean and non-numeric values in run comparisons

## 0.8.2 - 2026-09-14

- redact sensitive optimizer and scheduler parameters before constructing LLM context
- recognize sensitive names embedded in dotted, slashed, dashed, or spaced parameter paths
- preserve table rows containing triple dashes in HTML reports
- preserve unsupported PDF characters as explicit Unicode code-point escapes
- validate trace limits and return no events when `max_events=0`
- validate callback explanation intervals and ignore tensor-like boolean metrics

## 0.8.1 - 2026-09-14

- avoid no-op experiment recommendations for zero learning rates and zero-valued objectives
- recognize MAE, MAPE, MSE, MSLE, and RMSE as lower-is-better metrics
- preserve literal and escaped pipe characters in Markdown-to-HTML report tables
- inspect framework model references without evaluating ambiguous object truth values
- reject boolean, non-numeric, and invalid-step inputs in the real-time monitor

## 0.8.0 - 2026-08-17

- add a discoverable catalog of built-in prompts for scientific reports, improvement plans, training diagnosis, and experiment design
- allow report callers to parameterize prompt objectives, audiences, tone, rules, focus areas, headings, and return instructions
- add dependency-free real-time monitoring for non-finite metrics, stagnant losses, and possible overfitting
- add callback adapters shaped for Keras, Hugging Face Transformers, and PyTorch Lightning
- add structured, evidence-backed next-experiment recommendations and executable parameter mappings
- render and export next-experiment plans through the public report API
- document prompt discovery, live monitoring, callbacks, and controlled next-experiment workflows

## 0.7.0 - 2026-07-14

- ignore non-finite metrics in run comparisons to prevent invalid deltas and JSON values
- match complete metric-name tokens when inferring whether higher or lower values are better
- compact long metric histories while preserving endpoints, extrema, counts, and ordered samples
- avoid repeating metric-container literals in LLM notebook context
- allow LLM report callers to tune curve detail with `max_metric_points`
- recognize plain PyTorch model, optimizer, scheduler, and data-loader training parameters

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
