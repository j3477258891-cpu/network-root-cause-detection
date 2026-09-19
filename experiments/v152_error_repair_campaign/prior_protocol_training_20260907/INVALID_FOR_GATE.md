# Archived protocol, not usable for model selection

The previous partial run used validation-label-derived baseline F1 in action
ranking/eligibility. This violates the intended held-out action selection rule.
It also failed on an inner partition containing only one independent bin.

Preserved for audit only. All five folds must be rerun with label-blind decisions
and nested partitions that remain cross-fittable. None of these checkpoints may
authorize candidates, and their results are not leaderboard scores.
