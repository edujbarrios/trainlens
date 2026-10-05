# Agent Mode

TrainLens 0.16 adds a provider-free interface for coding agents and AutoResearch-style ML skills.
The goal is simple: **TrainLens supplies deterministic evidence; the surrounding agent supplies
reasoning.**

Agent Mode does not instantiate a model client, call an LLM endpoint, or require an API key.
Claude, Codex, Cursor, or another agent can use TrainLens as a local evidence layer while keeping
model access and tool execution inside the agent runtime that is already active.

## Why this exists

The direct LLM integration remains useful when a notebook wants TrainLens to request an
explanation itself. Coding agents are different: they already have a model, filesystem access,
and often a terminal. Calling a second model from TrainLens would duplicate credentials, cost,
and orchestration.

Agent Mode therefore exposes the deterministic boundary directly:

```text
training state -> TrainLens -> bounded evidence -> external agent
                                             <- structured plan
                  TrainLens <- evidence verification
```

## Python API

```python
from trainlens import build_agent_context

context = build_agent_context(globals())
```

`AgentContext` contains:

- bounded Markdown context derived from TrainLens analysis
- final metrics visible to the deterministic analyzer
- stable evidence IDs
- explicit instructions for controlled experiment design
- a JSON output schema for structured recommendations

It can be rendered for a human-readable agent prompt:

```python
print(context.to_markdown())
```

or serialized for tools:

```python
payload = context.to_json()
```

Dataset evidence can be attached explicitly, just like the existing LLM workflow:

```python
from trainlens import build_agent_context, explain_dataset

dataset_context = explain_dataset(
    {"train": train_dataset, "validation": validation_dataset},
    target="label",
)
context = build_agent_context(
    globals(),
    dataset_explanation=dataset_context,
)
```

## Verify the plan returned by the agent

The external agent should return JSON with a `recommendations` array. Every recommendation must
contain `action`, `rationale`, `evidence_ids`, `confidence`, and `success_criterion`.

```python
from trainlens import verify_agent_plan

plan = verify_agent_plan(agent_response, context=context)
print(plan.is_fully_supported)
print(plan.unsupported_evidence_ids)
```

Verification checks that cited evidence IDs exist in the deterministic context. This prevents an
agent from silently inventing TrainLens evidence references. It does **not** prove that the causal
interpretation or proposed experiment is correct.

## Portable-run CLI for skills

A separate terminal process cannot see the live memory of a Jupyter kernel. For coding-agent and
Skill workflows, save or capture a portable `TrainingRun` JSON and build context from that file:

```bash
trainlens agent-context candidate.json --format json > agent-context.json
```

The agent reads `agent-context.json`, produces `plan.json`, and can ask TrainLens to verify it:

```bash
trainlens verify-agent-plan plan.json --context agent-context.json
```

The context includes recorded run parameters and notes as citeable evidence IDs in addition to
metric-derived deterministic evidence.

## Recommended AutoResearch loop

A Skill can use TrainLens as one component of an iterative research loop:

1. inspect the current project and training evidence
2. build or refresh TrainLens agent context
3. identify the highest-value uncertainty
4. formulate one falsifiable hypothesis
5. propose the cheapest controlled experiment that tests it
6. cite TrainLens evidence IDs and define a measurable success criterion
7. execute the experiment with the project's existing training stack
8. capture the new run and compare it with prior evidence
9. stop when another experiment is not justified by expected information gain

Broad hyperparameter search should be a tool the agent can choose when evidence supports it, not
the default strategy.

See [`examples/agent-skill/SKILL.md`](../../examples/agent-skill/SKILL.md) for a small reusable
Skill template.

## Privacy and trust boundary

Agent Mode performs no network request. The serialized context is still data: review what your
external agent is allowed to read or transmit according to that agent's own privacy settings.
TrainLens keeps its existing context budgets and aggregate dataset protections, but it cannot
control what an external agent does after receiving the context.

The contract is deliberately narrow:

- TrainLens owns deterministic extraction, bounded context, evidence IDs, and verification.
- The external agent owns language-model reasoning and tool use.
- The human/research process owns scientific validity and final experiment decisions.
