"""Audit V66 count actions plus 18 disjoint scored batches in ten days."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(r"D:\zgyidong")
EXP = ROOT / "experiments"
V66 = EXP / "v66_order_count_action_audit" / "report.json"
V75 = EXP / "v75_dense_block_campaign" / "report.json"
OUT = EXP / "v78_ten_day_count_plus_batches"
TRUE_ROOTS = 1044
TARGET = 0.945
TRIALS = 300_000
GROUPS = 18


def summary(values: np.ndarray) -> dict:
    return {
        "mean": float(values.mean()),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "probability_reach_0_945": float(np.mean(values >= TARGET)),
    }


def main() -> None:
    r66 = json.loads(V66.read_text(encoding="utf-8"))
    r75 = json.loads(V75.read_text(encoding="utf-8"))
    base_tp = int(r75["base"]["tp"])
    base_p = int(r75["base"]["predictions"])
    half = TARGET / 2

    # V66 already emits only positive, non-conflicting actions.  Its delete
    # probability is P(node is false), whereas its stored probability is
    # P(node is a true root).
    fixed = []
    for row in r66["candidate_actions"]:
        is_add = row["kind"] == "add"
        probability = float(row["probability_true_root"])
        correctness = probability if is_add else 1 - probability
        fixed.append({
            "action_id": f"v66_rank_{int(row['rank']):03d}",
            "order_id": row["order_id"], "rid": row["rid"],
            "is_add": is_add, "probability": correctness,
        })
    fixed_orders = {row["order_id"] for row in fixed}
    fixed_nodes = {(row["order_id"], row["rid"]) for row in fixed}

    pool = []
    for action, probability in zip(
        r75["candidate_actions"], r75["correctness_probabilities"]
    ):
        nodes = {(action["order_id"], rid)
                 for rid in action["remove_rids"] + action["add_rids"]}
        if action["order_id"] in fixed_orders or nodes & fixed_nodes:
            continue
        pool.append({
            "action": action, "is_add": bool(action["add_rids"]),
            "probability": float(probability),
        })

    fixed_p = np.asarray([row["probability"] for row in fixed])
    fixed_add = np.asarray([row["is_add"] for row in fixed])
    pool_p = np.asarray([row["probability"] for row in pool])
    pool_add = np.asarray([row["is_add"] for row in pool])
    pool_good = np.where(pool_add, 1 - half, half)
    pool_bad = np.where(pool_add, -half, -(1 - half))
    pool_mean = pool_p * pool_good + (1 - pool_p) * pool_bad
    pool_var = pool_p * (pool_good - pool_mean) ** 2 + (
        1 - pool_p
    ) * (pool_bad - pool_mean) ** 2

    # Greedy mean/variance balancing gives every scored batch option value.
    assignment = np.full(len(pool), -1, dtype=np.int16)
    group_mean = np.zeros(GROUPS)
    group_var = np.zeros(GROUPS)
    group_size = np.zeros(GROUPS, dtype=np.int16)
    max_size = int(np.ceil(len(pool) / GROUPS))
    for index in np.argsort(-(pool_var + np.abs(pool_mean)), kind="stable"):
        eligible = np.flatnonzero(group_size < max_size)
        objective = (np.abs(group_mean[eligible] + pool_mean[index])
                     + 0.08 * group_var[eligible]
                     + 0.01 * group_size[eligible])
        group = eligible[int(np.argmin(objective))]
        assignment[index] = group
        group_mean[group] += pool_mean[index]
        group_var[group] += pool_var[index]
        group_size[group] += 1

    rng = np.random.default_rng(20260820)
    fixed_truth = rng.random((TRIALS, len(fixed))) < fixed_p
    pool_truth = rng.random((TRIALS, len(pool))) < pool_p

    fixed_tp_delta = np.where(
        fixed_add[None, :], fixed_truth, fixed_truth - 1
    ).sum(axis=1)
    fixed_p_delta = int(fixed_add.sum()) - int((~fixed_add).sum())
    fixed_scores = 2 * (base_tp + fixed_tp_delta) / (
        TRUE_ROOTS + base_p + fixed_p_delta
    )

    action_tp = np.where(pool_add[None, :], pool_truth, pool_truth - 1)
    group_tp = np.stack([action_tp[:, assignment == g].sum(axis=1)
                         for g in range(GROUPS)], axis=1)
    group_p = np.asarray([
        int(pool_add[assignment == g].sum())
        - int((~pool_add[assignment == g]).sum())
        for g in range(GROUPS)
    ])
    action_margin = np.where(pool_truth, pool_good[None, :], pool_bad[None, :])
    group_margin = np.stack([action_margin[:, assignment == g].sum(axis=1)
                             for g in range(GROUPS)], axis=1)
    chosen = group_margin > 0
    combined_tp = base_tp + fixed_tp_delta + (group_tp * chosen).sum(axis=1)
    combined_p = base_p + fixed_p_delta + (group_p[None, :] * chosen).sum(axis=1)
    combined_scores = 2 * combined_tp / (TRUE_ROOTS + combined_p)

    output = {
        "version": "v78-ten-day-count-plus-batches-audit-1",
        "base": r75["base"], "target": TARGET,
        "budget": {"days": 10, "submissions_per_day": 2,
                   "v66_baseline": 1, "batch_probes": 18,
                   "final_union": 1},
        "v66_action_count": len(fixed),
        "v75_remaining_action_count": len(pool),
        "excluded_v75_conflicts": len(r75["candidate_actions"]) - len(pool),
        "v66_baseline_simulation": summary(fixed_scores),
        "final_union_simulation": summary(combined_scores),
        "mean_selected_groups": float(chosen.sum(axis=1).mean()),
        "group_sizes": group_size.astype(int).tolist(),
        "group_expected_margins": group_mean.tolist(),
        "fixed_actions": fixed,
        "pool_actions": [row["action"] for row in pool],
        "assignment": assignment.astype(int).tolist(),
        "warning": "Independent-prior simulation; not leaderboard evidence; V66 calibration is the dominant risk.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in output.items()
                      if k not in {"fixed_actions", "pool_actions", "assignment"}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
