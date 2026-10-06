# Changelog

All notable changes to TrainLens will be documented here.

## 0.17.0 - 2026-10-06

- preserve full metric trajectories and step metadata when converting deterministic analysis into portable `TrainingRun` objects, so `Project.capture()` no longer collapses histories to final scalars
- keep train, validation, test, and custom split metrics distinct across run comparison, leaderboards, repeated-run analysis, checkpoint selection, and Agent Mode; ambiguous generic selectors now fail instead of silently choosing a split
- make portable run JSON stricter by rejecting boolean/non-finite metric values and emitting standards-compliant JSON without NaN/Infinity
- prevent pseudo-replication in `parameter_effects()` by matching repeated runs one-to-one and by seed when seed metadata is available
- require complete objective coverage before assigning ranks in multi-objective leaderboards and expose missing objectives explicitly
- make `AgentContext` mappings deeply immutable while keeping serialization round-trippable
- serialize `Project` writers with a cross-process lock file and atomic run-file replacement to avoid lost index updates during parallel experiment capture
- replace positional summary/signal/dataset evidence IDs with semantic content-derived identifiers that remain stable when findings are reordered

## 0.16.1 - 2026-10-05

- simplify the README around three clear workflows: local deterministic analysis, Agent Mode, and direct LLM reports
- move the end-to-end Colab quickstart notebook to the top of the README as the recommended starting point
- clarify that Agent Mode can use any LLM/provider already available to the surrounding agent, with Claude, Codex, Cursor, or another host normally owning that model connection
- make explicit that Agent Mode is optional and TrainLens can still call an OpenAI-compatible provider directly for generated reports, diagnoses, and improvement ideas
- refresh package, citation, badge, and release metadata for 0.16.1

## 0.16.0 - 2026-10-05

- add provider-free Agent Mode so Claude, Codex, Cursor, and other already-running agents can use TrainLens as a deterministic evidence layer without TrainLens-side API keys or model calls
- add public `AgentContext`, `build_agent_context()`, and `build_agent_context_from_run()` APIs with bounded Markdown/JSON context, stable evidence IDs, agent instructions, and a machine-readable output schema
- add `verify_agent_plan()` to validate structured external-agent recommendations against deterministic TrainLens evidence IDs
- add `trainlens agent-context` and `trainlens verify-agent-plan` for terminal-first AutoResearch and Skill workflows using portable `TrainingRun` JSON files
- include portable-run parameters and notes as citeable evidence while preserving existing metric-derived findings
- document the recommended evidence → hypothesis → controlled experiment → evidence loop and add a reusable `examples/agent-skill/SKILL.md` template
- keep the existing optional OpenAI-compatible provider workflow fully available and backwards compatible

## 0.15.1 - 2026-10-05

- extend `explain_dataset()` with dependency-light support for image and image-centered multimodal datasets
- recognize PIL-like images, image-shaped arrays/tensors, common image paths, Hugging Face-style image values, and multiple images per row without adding heavy runtime dependencies
- add `DatasetImageSummary`, `DatasetModalitySummary`, and `MultimodalDatasetExplanation` for aggregate visual and modality evidence
- summarize image counts, dimensions, aspect ratios, channels, modes, formats, images per row, and material cross-split image-size differences
- detect image+text and image+tabular splits as multimodal while preserving existing behavior for non-visual datasets
- keep visual context privacy-preserving by excluding pixels, image bytes, private paths, and filenames from rendered or outbound dataset evidence
- verify image features and visual observations participate in stable evidence IDs used by structured improvement plans
- document image and multimodal workflows in the README, dataset guide, and a dependency-free runnable example

## 0.15.0 - 2026-10-05

- add public `explain_dataset()` for local, deterministic dataset explanations without adding pandas or Hugging Face runtime dependencies
- summarize split sizes, feature kinds, missingness, cardinality, numeric ranges and means, average text lengths, supervised target distributions, and continuous-target ranges
- compare named splits for feature-schema differences and material target-distribution changes while keeping observations explicitly descriptive rather than causal
- keep dataset profiles aggregate-only so raw feature rows are not included in rendered explanations
- allow an explicit `DatasetExplanation` to be attached to notebook-context previews, paper reports, improvement plans, and structured verified plans
- keep dataset context bounded by `ContextPolicy` and never silently add the underlying raw dataset to an optional LLM request
- add stable evidence IDs for dataset splits, features, targets, and observations so structured LLM recommendations can cite dataset facts and unsupported citations remain detectable
- expand the README and documentation around dataset-aware model interpretation, including privacy and causal-interpretation limits
- update the Colab quickstart to demonstrate dataset explanation, training diagnosis, request preview, and evidence-backed next-experiment planning end to end

## 0.14.3 - 2026-10-05

- add a local-first `Project` store for portable training runs, names, and tags under `.trainlens`
- add multi-run leaderboards with objectives, hard constraints, Pareto-front marking, and notebook-friendly Markdown rendering
- add repeated-run aggregation and conservative group comparisons so seed-to-seed variability is visible before treating small metric changes as meaningful
- add `parameter_effects()` to summarize observed changes only from controlled run pairs that differ in exactly one recorded parameter
- add deterministic validation-based checkpoint selection with best-step evidence, late degradation, suggested patience, and explicit rejection of held-out test metrics for tuning
- add reproducibility provenance for Git state, seed, Python/platform, selected package versions, and deterministic dataset fingerprints
- add portable duration, throughput, memory, GPU-memory, and estimated-cost metrics that can participate in comparisons and Pareto objectives
- add the `trainlens compare` and `trainlens check` CLI commands for portable run comparisons, material-regression failures, and repeatable metric gates in CI
- expose the analyzer registry as a public plugin API with per-analysis selection through `AnalysisConfig`
- add structured optional LLM improvement plans with stable deterministic evidence IDs and explicit reporting of unsupported citations
- substantially shorten and reposition the README around experiment evidence and next-run decisions, with the new tagline “TrainLens turns training runs into evidence: what changed, what went wrong, and what experiment to run next.”

## 0.14.2 - 2026-10-04

- add public `load_prompt()` so reusable prompt instructions can live in UTF-8 `.md` files instead of being embedded in Python or notebook cells
- preserve Markdown structure and relative indentation when loading prompt files while removing only surrounding whitespace
- reject non-`.md` paths explicitly so accidental file loads fail early and prompt files remain easy to identify and version
- add `examples/prompts/improvement_plan.md` plus README and Colab quickstart examples for file-based prompt authoring
- add focused tests for Markdown loading through both string and `Path` inputs, case-insensitive `.md` suffixes, formatting preservation, and invalid extensions
- update the README release badge for 0.14.2

## 0.14.1 - 2026-10-04

- solve the prompt-authoring friction caused by splitting long LLM instructions into many adjacent quoted strings: add public `prompt_text()` so users can write normal indented triple-quoted text and have TrainLens remove common code indentation and surrounding blank space before use
- preserve relative indentation inside multiline prompt text, keeping lists and nested structure readable
- update the README and Colab quickstart to use the new multiline prompt workflow
- add focused tests for multiline dedenting and whitespace cleanup
- allow releases to use version-specific notes files so the GitHub release can explain the user-facing problem and solution instead of relying only on generated PR titles

## 0.14.0 - 2026-10-03

- make train, validation, and held-out test first-class metric splits, including split-aware metric containers and scalar test metrics
- promote final fine-tuning metrics such as F1, precision, recall, and perplexity into structured analysis results
- make train/validation overfitting detection direction-aware, including loss-only training histories, and add cautious validation-to-test degradation diagnostics
- recommend a final held-out test evaluation when validation results exist but test results are missing
- add `training_curves()` and optional `plot_training_curves()` through the new `trainlens[plots]` extra
- replace the minimal README/Colab entry point with a realistic fine-tuning walkthrough covering plots, local diagnosis, safe LLM API-key entry, exact prompt preview, explanations, and evidence-backed next experiments
- document train/validation/test workflows and plotting in English and Spanish

## 0.13.0 - 2026-10-03

- add `OpenAICompatibleProvider.from_values()` and `from_env()` convenience constructors for shorter remote and local LLM setup
- add `LLMRequestPreview`, `OpenAICompatibleProvider.preview()`, and notebook-level `preview_llm_request()` so users can inspect the exact system prompt, sanitized evidence, endpoint, model, and JSON payload before any network request
- make the real OpenAI-compatible request reuse the same preview builder, keeping inspected prompts and on-wire payloads aligned while excluding credentials from previews
- strengthen the reusable improvement-plan prompt so every recommendation includes supporting evidence, a concrete change, rationale, expected effect, cost or risk, confidence, and a measurable success criterion, followed by one controlled next run
- expose common scientific, diagnosis, improvement, and controlled-experiment prompt recipes directly from the top-level `trainlens` package
- document an end-to-end LLM improvement workflow in the README and English/Spanish guides, including remote/local provider loading, prompt definition, request preview, and report generation

## 0.12.0 - 2026-10-03

- make `RunComparison` render itself as Markdown in Jupyter and expose `to_markdown()` for the same notebook-friendly representation in regular Python
- replace the README's configuration-heavy first example with a two-cell, local-only comparison quickstart that requires no LLM provider or API key
- add a Colab-ready `examples/quickstart.ipynb` so new users can try TrainLens from the README with minimal setup
- refresh project and citation metadata around TrainLens' notebook-first, local-analysis workflow

## 0.11.0 - 2026-10-01

- add public `AnalysisConfig` controls for explicit model/trainer selection, strict ambiguity handling, and authoritative metric/label evidence across deterministic analysis and LLM context
- ground optional LLM reports in the local `AnalysisResult`, including deterministic summaries, signals, evidence provenance, and recommendations rather than asking the provider to reinterpret notebook state independently
- expand `TrainingProfile` with partial fine-tuning detection, trainable/total parameter counts and fractions, multiple/active PEFT adapters, and 4-bit/8-bit quantization metadata while remaining framework-dependency-free
- add an extensible `MetricRegistry` with aliases, bounds, units, and material-change thresholds, plus Pareto-front and constrained multi-objective experiment planning
- make training runs portable with JSON `save_run()` / `load_run()` helpers and richer comparisons covering changed parameters, best trajectory values, and observation counts
- add custom live-monitoring detectors with explicit persistent-alert lifecycle support while preserving deterministic built-in stagnation and overfitting alerts
- add public `LLMProvider` injection, expose the OpenAI-compatible provider/configuration APIs, and allow local unauthenticated endpoints without a placeholder API key
- add public `ContextPolicy` budgets for metric series, notebook variables, training artifacts, model candidates, and total outbound context size with explicit truncation notices

## 0.10.1 - 2026-10-01

- report material metric changes even when optimization direction is unknown and preserve common 0–100 percentage scales in experiment success criteria
- avoid no-op or backwards weight-decay recommendations when regularization is already configured
- reuse one notebook snapshot in Python report builders and keep Hugging Face trainer configuration aligned with the selected model
- recognize slash-, dash-, and space-separated loss aliases during live overfitting monitoring
- emit callback explanations at most once per training step when frameworks provide repeated updates

## 0.10.0 - 2026-09-29

- add public `TrainingProfile` and `inspect_training_profile()` APIs for dependency-light training-configuration introspection
- understand common Hugging Face training arguments, PEFT/LoRA/DoRA settings, VLM projector configuration, and multimodal learning-rate splits through safe duck typing
- detect projector-only alignment, parameter-efficient VLM fine-tuning, and full VLM fine-tuning while reporting frozen and trainable multimodal components
- add VLM-specific signals and controlled recommendations for very small adapter ranks, fixed multimodal sequence lengths, split projector learning rates, and accidentally frozen training configurations
- include the normalized VLM/PEFT training profile in deterministic notebook analysis and the exact sanitized context previewed or sent to an optional LLM provider
- keep the new VLM training intelligence dependency-free with respect to Transformers, PEFT, and PyTorch

## 0.9.2 - 2026-09-28

- keep notebook analysis best-effort when metric histories, model configuration, array metadata, or feature-importance properties raise during inspection
- avoid consuming one-shot label iterators while automatically checking notebook data for class imbalance
- coalesce repeated real-time monitoring callbacks at the same training step so they do not advance patience windows or create premature stagnation/overfitting alerts
- reuse one captured notebook snapshot for deterministic local analysis and outbound LLM evidence in `%explain_training`
- cap OpenAI-compatible provider response bodies at 4 MiB and report oversized or invalid UTF-8 responses explicitly
- enforce an 80% coverage floor and continuously test the declared minimum IPython, Jinja2, and Rich runtime dependencies

## 0.9.1 - 2026-09-28

- reject boolean and non-numeric `minimum_improvement` values at the next-experiment recommendation boundary instead of silently treating booleans as numbers or leaking low-level type errors
- ignore framework callback updates that contain no usable numeric metrics so metadata-only events do not distort monitoring windows or re-arm persistent alerts
- verify project and public package versions agree before creating a release tag
- install and smoke-test the freshly built wheel in an isolated environment before Trusted Publishing can upload it to PyPI

## 0.9.0 - 2026-09-25

- preserve metric history integrity with aligned step metadata, finite-value filtering, fractional epochs, and bounded one-dimensional array-like histories
- centralize metric optimization semantics and natural bounds across run comparison and next-experiment recommendations
- add persistent alert lifecycle handling so stagnation and overfitting alerts re-arm only after recovery
- add optional framework-native callback adapters for Keras 3, Hugging Face Transformers, and PyTorch Lightning while retaining the dependency-free callback core
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
