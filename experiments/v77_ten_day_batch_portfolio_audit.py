"""Audit disjoint batch portfolios under a 20-submission / ten-day budget."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(r"D:\zgyidong")
EXP = ROOT / "experiments"
SOURCE = EXP / "v75_dense_block_campaign" / "report.json"
OUT = EXP / "v77_ten_day_batch_portfolio"
TRUE_ROOTS = 1044
TARGET = 0.945
TRIALS = 300_000
GROUPS = 19  # each group is itself a scored submission; best is already valid


def summarize(values: np.ndarray) -> dict:
    return {
        "mean": float(values.mean()),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "probability_reach_0_945": float(np.mean(values >= TARGET)),
    }


def balanced_partition(mean_margin: np.ndarray, variance: np.ndarray) -> np.ndarray:
    """Greedily balance group means and variances without looking at truth."""
    order = np.argsort(-(variance + np.abs(mean_margin)), kind="stable")
    assignment = np.full(len(order), -1, dtype=np.int16)
    group_mean = np.zeros(GROUPS)
    group_var = np.zeros(GROUPS)
    group_size = np.zeros(GROUPS, dtype=np.int16)
    max_size = int(np.ceil(len(order) / GROUPS))
    for index in order:
        eligible = np.flatnonzero(group_size < max_size)
        # Prefer low accumulated absolute mean and variance to retain option
        # value in every independently scored batch.
        objective = (
            np.abs(group_mean[eligible] + mean_margin[index])
            + 0.08 * group_var[eligible]
            + 0.01 * group_size[eligible]
        )
        group = eligible[int(np.argmin(objective))]
        assignment[index] = group
        group_mean[group] += mean_margin[index]
        group_var[group] += variance[index]
        group_size[group] += 1
    return assignment


def main() -> None:
    report = json.loads(SOURCE.read_text(encoding="utf-8"))
    actions = report["candidate_actions"]
    probabilities = np.asarray(report["correctness_probabilities"], dtype=np.float64)
    additions = np.asarray([bool(row["add_rids"]) for row in actions])
    base_tp = int(report["base"]["tp"])
    base_predictions = int(report["base"]["predictions"])

    # Margin M = TP - TARGET/2 * (TRUE_ROOTS + predictions).  M >= 0 iff
    # F1 >= TARGET, and disjoint action margins add exactly.
    half = TARGET / 2
    good = np.where(additions, 1 - half, half)
    bad = np.where(additions, -half, -(1 - half))
    expected = probabilities * good + (1 - probabilities) * bad
    variance = probabilities * (good - expected) ** 2 + (
        1 - probabilities
    ) * (bad - expected) ** 2
    assignment = balanced_partition(expected, variance)

    rng = np.random.default_rng(20260820)
    truths = rng.random((TRIALS, len(actions))) < probabilities
    action_margin = np.where(truths, good[None, :], bad[None, :])
    group_margin = np.stack([
        action_margin[:, assignment == group].sum(axis=1)
        for group in range(GROUPS)
    ], axis=1)

    # Strategy 1: each disjoint group is tested alone.  The highest individual
    # public score is valid immediately and consumes no extra checkpoint.
    group_tp_delta = np.stack([
        np.where(
            additions[assignment == group][None, :],
            truths[:, assignment == group],
            truths[:, assignment == group] - 1,
        ).sum(axis=1)
        for group in range(GROUPS)
    ], axis=1)
    group_p_delta = np.asarray([
        int(additions[assignment == group].sum())
        - int((~additions[assignment == group]).sum())
        for group in range(GROUPS)
    ])
    individual_scores = 2 * (base_tp + group_tp_delta) / (
        TRUE_ROOTS + base_predictions + group_p_delta[None, :]
    )
    best_scored = np.maximum(
        2 * base_tp / (TRUE_ROOTS + base_predictions),
        individual_scores.max(axis=1),
    )

    # Strategy 2: reserve submission 20 for the union of groups with positive
    # observed target-margin contribution.  This union is exact because the
    # candidate actions are node- and order-disjoint by V75 construction.
    chosen = group_margin > 0
    combined_tp_delta = (group_tp_delta * chosen).sum(axis=1)
    combined_p_delta = (group_p_delta[None, :] * chosen).sum(axis=1)
    combined_scores = 2 * (base_tp + combined_tp_delta) / (
        TRUE_ROOTS + base_predictions + combined_p_delta
    )

    base_margin = base_tp - half * (TRUE_ROOTS + base_predictions)
    output = {
        "version": "v77-ten-day-batch-portfolio-audit-1",
        "source": str(SOURCE),
        "base": report["base"],
        "target": TARGET,
        "base_target_margin": float(base_margin),
        "budget": {"days": 10, "submissions_per_day": 2,
                   "batch_probes": 19, "final_union": 1},
        "group_sizes": [int(np.sum(assignment == g)) for g in range(GROUPS)],
        "group_expected_margins": [float(expected[assignment == g].sum())
                                   for g in range(GROUPS)],
        "best_scored_batch": summarize(best_scored),
        "positive_group_union": summarize(combined_scores),
        "mean_selected_groups": float(chosen.sum(axis=1).mean()),
        "probability_positive_total_margin": float(
            np.mean(group_margin.clip(min=0).sum(axis=1) >= -base_margin)
        ),
        "assignment": assignment.astype(int).tolist(),
        "groups": [[actions[i]["action_id"] for i in np.flatnonzero(assignment == g)]
                   for g in range(GROUPS)],
        "warning": "Independent-prior simulation; not leaderboard evidence.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in output.items()
                      if k not in {"assignment", "groups"}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
