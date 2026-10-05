# Experiment history

TrainLens 0.14.3 adds local-first tools for treating training runs as an experimental history rather than isolated notebook outputs.

Use `Project` to persist portable runs locally, `leaderboard()` to rank multiple runs, `aggregate_runs()` and `compare_run_groups()` to reason about repeated seeds, and `parameter_effects()` to summarize controlled one-parameter changes observed in your own experiments.

These helpers remain dependency-free and operate on `TrainingRun` objects, so they can be combined with existing JSON import/export, Pareto-front planning, and comparison APIs without introducing a tracking service.
