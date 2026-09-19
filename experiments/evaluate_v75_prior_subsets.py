"""Compare unverified direct candidate subsets against the decoded V75 path."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(r"D:\zgyidong")
EXP = ROOT / "experiments"
BASE_TP, BASE_P, TRUE_ROOTS, TARGET = 956, 1035, 1044, 0.95


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    reports = [read(EXP / "v58_coded_campaign/report.json"),
               read(EXP / "v59_extended_coded_campaign/report.json")]
    actions = [item for report in reports for item in report["candidate_actions"]]
    probabilities = np.asarray(
        [item for report in reports for item in report["correctness_probabilities"]],
        dtype=np.float64,
    )
    additions = np.asarray([bool(item["add_rids"]) for item in actions])
    ranking = np.argsort(-probabilities * np.where(additions, 1.05, 0.95), kind="stable")
    rng = np.random.default_rng(20260821)
    truths = rng.random((300_000, len(actions))) < probabilities
    rows = []
    for count in (20, 40, 60, 80, 100, 125, 150, 175, 200, 225, 250, 275, 290):
        indices = ranking[:count]
        add = additions[indices]
        labels = truths[:, indices]
        tp = BASE_TP + (labels & add).sum(axis=1) - ((~labels) & (~add)).sum(axis=1)
        predictions = BASE_P + int(add.sum()) - int((~add).sum())
        scores = 2 * tp / (TRUE_ROOTS + predictions)
        rows.append({
            "actions": count, "adds": int(add.sum()), "deletes": int((~add).sum()),
            "predictions": predictions, "expected_correctness": float(probabilities[indices].mean()),
            "mean": float(scores.mean()), "p05": float(np.quantile(scores, 0.05)),
            "median": float(np.median(scores)), "p95": float(np.quantile(scores, 0.95)),
            "probability_reach_0_94": float(np.mean(scores >= 0.94)),
            "probability_reach_0_95": float(np.mean(scores >= TARGET)),
        })
    # This is the different, intended V75 mode: labels are first recovered by
    # score probes, then only correct actions are placed in a checkpoint.
    # It estimates when a checkpoint is likely to become worthwhile, but is
    # still a prior simulation rather than a public-score prediction.
    selected_report = read(EXP / "v75_dense_block_campaign/report.json")
    selected_actions = selected_report["candidate_actions"]
    selected_probabilities = np.asarray(selected_report["correctness_probabilities"])
    selected_additions = np.asarray([bool(item["add_rids"]) for item in selected_actions])
    selected_truths = rng.random((300_000, len(selected_actions))) < selected_probabilities
    decoded_rows = []
    for block_count in range(1, 6):
        end = block_count * 35
        labels = selected_truths[:, :end]
        add = selected_additions[:end]
        correct_adds = (labels & add).sum(axis=1)
        correct_deletes = (labels & ~add).sum(axis=1)
        predictions = BASE_P + correct_adds - correct_deletes
        scores = 2 * (BASE_TP + correct_adds) / (TRUE_ROOTS + predictions)
        decoded_rows.append({
            "decoded_blocks": block_count,
            "candidate_actions": end,
            "mean": float(scores.mean()), "p05": float(np.quantile(scores, 0.05)),
            "median": float(np.median(scores)), "p95": float(np.quantile(scores, 0.95)),
            "probability_reach_0_94": float(np.mean(scores >= 0.94)),
            "probability_reach_0_95": float(np.mean(scores >= TARGET)),
        })
    print(json.dumps({
        "warning": "Independent-prior simulation only; no row is public leaderboard evidence.",
        "candidate_count": len(actions),
        "rows": rows,
        "decoded_checkpoint_rows": decoded_rows,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
