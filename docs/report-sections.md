# Report Sections

TrainLens reports are designed to explain the result of a training run, not just
print metric values.

## Training Summary

This section lists the model, framework, loss or accuracy movement, detected
foundation-model profile, and other high-level facts from the notebook state.

## Result Explanation

This section interprets the result. For example, TrainLens compares training and
validation loss, calls out large gaps, counts warning signals, and notes whether
execution trace evidence is available.

## Metrics And Trace

Metric tables show the latest normalized values. Execution trace tables connect
those values to concrete events such as batches, eval steps, and checkpoints.

## Potential Issues

Signals are heuristic findings such as validation instability, loss plateaus,
adapter under-capacity, contrastive misalignment, or class imbalance.

## Recommended Next Steps

Recommendations are specific actions with rationale. They are intended to be
small enough to try in the next experiment.

## Improvement Plan

The improvement plan sorts recommendations by confidence and explains why each
step matters. This is the section to follow when deciding what to change next.
