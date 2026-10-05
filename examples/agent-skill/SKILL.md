# TrainLens ML Research Skill

Use TrainLens as the deterministic evidence layer for ML experiment decisions. The coding agent
running this Skill is the reasoning runtime; do not configure or call a second LLM through
TrainLens.

## Protocol

1. Find the current portable TrainLens `TrainingRun` JSON, or create one with the project's normal
   TrainLens workflow.
2. Run `trainlens agent-context <run.json> --format json --output agent-context.json`.
3. Read `agent-context.json` completely before proposing a new experiment.
4. Identify the single highest-value uncertainty in the current evidence.
5. Form one falsifiable hypothesis. Distinguish the hypothesis from observed evidence.
6. Prefer the cheapest controlled experiment that can test the hypothesis. Change one important
   variable at a time unless a coupled change is technically necessary.
7. Never tune against held-out test evidence. Use test results only as final generalization
   evidence.
8. Return a JSON object matching the `output_schema` in `agent-context.json`. Cite TrainLens
   `evidence_id` values for every recommendation and define a measurable success criterion.
9. Save the response as `plan.json` and run
   `trainlens verify-agent-plan plan.json --context agent-context.json`.
10. Do not execute a recommendation that fails evidence verification. If it passes, make the
    smallest necessary code/config change, run the experiment, capture the new evidence, and
    repeat only when another experiment is justified.

## Research policy

- Treat TrainLens findings as evidence, not proof of causation.
- Prefer information gain over blind metric search.
- Use HPO or AutoML only when evidence indicates parameter search is the current bottleneck.
- Preserve reproducibility metadata and recorded parameters for every run.
- Stop when improvements are within observed run-to-run variability or when the next experiment
  is not worth its expected cost.
