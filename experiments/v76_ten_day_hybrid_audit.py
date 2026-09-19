"""Audit ten-day hybrid strategies for reaching public F1 >= 0.945.

The public leaderboard allows at most two submissions per day in the current
campaign assumption.  A ten-day plan can therefore spend at most 19 probes
and must reserve one submission for the decoded checkpoint.  This audit asks
whether using those probes to decode the highest *decision-value* actions,
while directly applying sufficiently strong-prior actions, can meet the goal.

This is a prior-sampled feasibility audit, not public leaderboard evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(r"D:\zgyidong")
EXP = ROOT / "experiments"
SOURCE = EXP / "v75_dense_block_campaign" / "report.json"
OUT = EXP / "v76_ten_day_hybrid"
TRUE_ROOTS = 1044
TARGET = 0.945
TRIALS = 300_000


def score(tp: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    return 2.0 * tp / (TRUE_ROOTS + predictions)


def apply_policy(
    truths: np.ndarray,
    additions: np.ndarray,
    decoded: np.ndarray,
    direct: np.ndarray,
    base_tp: int,
    base_predictions: int,
) -> np.ndarray:
    chosen = direct[None, :] | (decoded[None, :] & truths)
    correct_adds = (chosen & truths & additions[None, :]).sum(axis=1)
    wrong_adds = (chosen & ~truths & additions[None, :]).sum(axis=1)
    correct_deletes = (chosen & truths & ~additions[None, :]).sum(axis=1)
    wrong_deletes = (chosen & ~truths & ~additions[None, :]).sum(axis=1)
    tp = base_tp + correct_adds - wrong_deletes
    predictions = (
        base_predictions + correct_adds + wrong_adds
        - correct_deletes - wrong_deletes
    )
    return score(tp, predictions)


def summarize(values: np.ndarray) -> dict:
    return {
        "mean": float(values.mean()),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "probability_reach_0_945": float(np.mean(values >= TARGET)),
    }


def main() -> None:
    report = json.loads(SOURCE.read_text(encoding="utf-8"))
    actions = report["candidate_actions"]
    probabilities = np.asarray(report["correctness_probabilities"], dtype=np.float64)
    additions = np.asarray([bool(row["add_rids"]) for row in actions])
    n = len(actions)
    base_tp = int(report["base"]["tp"])
    base_predictions = int(report["base"]["predictions"])

    # Exact one-action score deltas define both the direct decision and the
    # incremental value of learning the action label before the checkpoint.
    base_score = 2 * base_tp / (TRUE_ROOTS + base_predictions)
    expected_direct_gain = np.zeros(n)
    oracle_gain = np.zeros(n)
    for i, (p, is_add) in enumerate(zip(probabilities, additions)):
        if is_add:
            good = 2 * (base_tp + 1) / (TRUE_ROOTS + base_predictions + 1)
            bad = 2 * base_tp / (TRUE_ROOTS + base_predictions + 1)
        else:
            good = 2 * base_tp / (TRUE_ROOTS + base_predictions - 1)
            bad = 2 * (base_tp - 1) / (TRUE_ROOTS + base_predictions - 1)
        expected_direct_gain[i] = p * good + (1 - p) * bad - base_score
        oracle_gain[i] = p * (good - base_score)
    direct = expected_direct_gain > 0
    decision_value = oracle_gain - np.maximum(expected_direct_gain, 0)

    rng = np.random.default_rng(20260820)
    truths = rng.random((TRIALS, n)) < probabilities[None, :]
    rows = []
    for decoded_count in (0, 25, 30, 35):
        decoded_indices = np.argsort(-decision_value, kind="stable")[:decoded_count]
        decoded = np.zeros(n, dtype=bool)
        decoded[decoded_indices] = True
        use_direct = direct & ~decoded
        values = apply_policy(
            truths, additions, decoded, use_direct, base_tp, base_predictions
        )
        rows.append({
            "decoded_count": decoded_count,
            "direct_count": int(use_direct.sum()),
            "decoded_adds": int((decoded & additions).sum()),
            "decoded_deletes": int((decoded & ~additions).sum()),
            "summary": summarize(values),
            "decoded_actions": [actions[i]["action_id"] for i in decoded_indices],
        })

    oracle = apply_policy(
        truths, additions, np.ones(n, dtype=bool), np.zeros(n, dtype=bool),
        base_tp, base_predictions,
    )
    output = {
        "version": "v76-ten-day-hybrid-audit-1",
        "source": str(SOURCE),
        "base": report["base"],
        "budget": {"days": 10, "submissions_per_day": 2, "max_probes": 19,
                   "reserved_final_checkpoint": 1},
        "candidate_count": n,
        "direct_policy_count": int(direct.sum()),
        "rows": rows,
        "full_oracle": summarize(oracle),
        "warning": "Independent-prior simulation; not leaderboard evidence.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "base": output["base"],
        "budget": output["budget"],
        "direct_policy_count": output["direct_policy_count"],
        "rows": [{k: v for k, v in row.items() if k != "decoded_actions"}
                 for row in rows],
        "full_oracle": output["full_oracle"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
