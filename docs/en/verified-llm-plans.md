# Verified LLM improvement plans

`build_verified_improvement_plan()` adds a machine-readable path for optional LLM recommendations. TrainLens first builds deterministic local evidence IDs, sends those IDs with the sanitized report, parses the returned JSON plan, and verifies every citation against the evidence catalog.

```python
from trainlens import build_verified_improvement_plan

plan = build_verified_improvement_plan(globals(), provider=provider)

for recommendation in plan.recommendations:
    print(recommendation.action)
    print(recommendation.evidence_ids)
    print(recommendation.is_supported)

print(plan.unsupported_evidence_ids)
```

A recommendation is only marked supported when it cites at least one supplied evidence ID and every cited ID exists. Unknown or invented IDs are preserved in `unsupported_evidence_ids` rather than silently accepted. This verifies citation provenance; it does not prove that an LLM's causal interpretation is correct.

Use `evidence_catalog(result)` and `parse_verified_improvement_plan(response, evidence=...)` separately when you want to manage the LLM request yourself.
