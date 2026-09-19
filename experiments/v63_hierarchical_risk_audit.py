"""Audit V58+V59 with correlated Beta-Binomial calibration uncertainty."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
V40 = EXP / "v40_one_sided_boundary/report.json"
V58 = EXP / "v58_coded_campaign/report.json"
V59 = EXP / "v59_extended_coded_campaign/report.json"
OUT = EXP / "v63_hierarchical_risk"
TRUE_ROOTS = 1044
BASE_TP = 956
BASE_P = 1035
TARGET = 0.95


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def v40_increment_bins():
    report = read_json(V40)
    bins = {}
    for kind in ("add", "delete"):
        previous_k = previous_correct = 0
        for row in report["oof_curves"][kind]:
            k, correct = int(row["k"]), int(row["correct"])
            bins[(kind, k)] = {
                "start": previous_k + 1, "stop": k,
                "correct": correct - previous_correct,
                "total": k - previous_k,
            }
            previous_k, previous_correct = k, correct
    return bins


def group_for_action(action, bins):
    evidence = action.get("evidence", {})
    source = action["source"]
    if source == "v40_boundary_frontier":
        kind = "add" if action["add_rids"] else "delete"
        rank = int(evidence["source_rank"])
        candidates = [(stop, row) for (local_kind, stop), row in bins.items()
                      if local_kind == kind and rank <= stop]
        if not candidates:
            stop = max(key[1] for key in bins if key[0] == kind)
            row = bins[(kind, stop)]
        else:
            stop, row = min(candidates)
        return ("v40", kind, stop), row["correct"], row["total"]
    if source in ("v50_empirical_bayes_stratum", "v48_station_stable_stratum"):
        rule = evidence["matched_rule"]
        fields = tuple(rule.get("fields", []))
        values = tuple(str(value) for value in rule.get("values", []))
        support = int(rule["support"])
        correct = int(rule["correct"])
        return (source, fields, values), correct, support
    # Count-model margins are not calibrated probabilities; use a weak prior
    # centered at the previously derived correctness estimate.
    score = float(evidence.get("node_score", 0.5))
    probability = score if action["add_rids"] else 1 - score
    strength = 8
    return (source, action["action_id"]), probability * strength, strength


def simulate(actions, trials=100000, seed=20260827, stress=1.0):
    bins = v40_increment_bins()
    grouped = defaultdict(list)
    parameters = {}
    for index, action in enumerate(actions):
        key, correct, total = group_for_action(action, bins)
        grouped[key].append(index)
        parameters[key] = (float(correct), float(total))
    rng = np.random.default_rng(seed)
    scores = np.empty(trials, dtype=np.float64)
    reached = 0
    for trial in range(trials):
        correct_mask = np.zeros(len(actions), dtype=bool)
        for key, indices in grouped.items():
            correct, total = parameters[key]
            alpha, beta = 1 + correct, 1 + total - correct
            probability = float(rng.beta(alpha, beta))
            # Stress below 1 shrinks log-odds toward an adverse interpretation:
            # positive rates are multiplied, capped to preserve [0,1].
            probability = min(1.0, max(0.0, probability * stress))
            correct_mask[indices] = rng.random(len(indices)) < probability
        true_adds = sum(flag and bool(action["add_rids"])
                        for flag, action in zip(correct_mask, actions))
        delete_count = sum(bool(action["remove_rids"]) for action in actions)
        correct_deletes = sum(flag and bool(action["remove_rids"])
                              for flag, action in zip(correct_mask, actions))
        incorrect_deletes = delete_count - correct_deletes
        tp = BASE_TP + true_adds - incorrect_deletes
        predictions = BASE_P + sum(bool(action["add_rids"]) for action in actions) - delete_count
        score = 2 * tp / (TRUE_ROOTS + predictions)
        scores[trial] = score
        reached += int(score >= TARGET)
    scores.sort()
    return {
        "trials": trials, "stress_multiplier": stress,
        "group_count": len(grouped), "mean": float(scores.mean()),
        "p01": float(scores[int(0.01 * trials)]),
        "p05": float(scores[int(0.05 * trials)]),
        "median": float(scores[trials // 2]),
        "p95": float(scores[int(0.95 * trials)]),
        "probability_reach_0_95": reached / trials,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    actions = (read_json(V58)["candidate_actions"] +
               read_json(V59)["candidate_actions"])
    scenarios = {
        "hierarchical_calibrated": simulate(actions, stress=1.0),
        "hierarchical_10pct_adverse": simulate(actions, seed=20260828, stress=0.9),
        "hierarchical_20pct_adverse": simulate(actions, seed=20260829, stress=0.8),
    }
    report = {
        "version": "v63-hierarchical-risk-audit-1",
        "candidate_count": len(actions), "target": TARGET,
        "model": "Shared Beta-Binomial rates by OOF rank band or matched rule.",
        "scenarios": scenarios,
        "interpretation": "This models correlated calibration error; it is still a statistical audit, not a leaderboard guarantee.",
    }
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
