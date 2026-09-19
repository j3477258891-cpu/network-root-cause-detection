"""Build a 40-query quantitative code over 129 disjoint candidate actions.

The action pool combines V49 (4), V50 (61), and V52 (64).  For an addition,
the hidden correctness bit is the alarm label.  For a deletion, it is one
minus the alarm label.  A leaderboard TP delta for any coded subset therefore
reveals the exact sum of its correctness bits after adding back the number of
included deletions.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
V30 = EXP / "v30_meta_stack"
for value in (EXP, V30):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from build_cross_order_probes import apply_actions, load_submission, write_submission
from v37_online_equation_solver import CHAMPION, TRUE_ROOTS


V37 = EXP / "v37_online_equations"
V49 = EXP / "v49_extended_count_batch"
V50 = EXP / "v50_disjoint_batches"
V52 = EXP / "v52_precision_frontier"
V40 = EXP / "v40_one_sided_boundary/report.json"
RECORDS = EXP / "v25_semantic_router/cloud_dataset/semantic_records.json.gz"
OUT = EXP / "v58_coded_campaign"
BASE_TP = 956
BASE_P = 1035
ROWS = 60
TARGET = 0.95


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def action_key(action):
    return (action["order_id"], tuple(action["remove_rids"]), tuple(action["add_rids"]))


def collect_actions():
    v49 = [action for action in read_json(V49 / "report.json")["manifest"]["actions"]
           if action["source"] != "historical_leaderboard_equations"]
    v50, seen = [], set()
    for manifest in read_json(V50 / "report.json")["probes"].values():
        for action in manifest["actions"]:
            if action["source"] != "v50_empirical_bayes_stratum" or action_key(action) in seen:
                continue
            seen.add(action_key(action))
            v50.append(action)
    v52, seen52 = [], set()
    for manifest in read_json(V52 / "report.json")["probes"].values():
        for action in manifest["actions"]:
            if action["source"] == "historical_leaderboard_equations" or action_key(action) in seen52:
                continue
            seen52.add(action_key(action))
            v52.append(action)
    actions = v49 + v50 + v52
    if (len(v49), len(v50), len(v52), len(actions)) != (4, 61, 64, 129):
        raise RuntimeError((len(v49), len(v50), len(v52), len(actions)))
    orders = [action["order_id"] for action in actions]
    nodes = [(action["order_id"], rid) for action in actions
             for rid in action["remove_rids"] + action["add_rids"]]
    if len(set(orders)) != 129 or len(set(nodes)) != 129:
        raise RuntimeError("V58 actions are not order/node disjoint")
    return actions, {"v49": 4, "v50": 61, "v52": 64}


def correctness_probability(action):
    evidence = action.get("evidence", {})
    if action["source"] == "v50_empirical_bayes_stratum":
        return float(evidence["matched_rule"]["posterior_mean"])
    if action["source"] == "v48_station_stable_stratum":
        return float(evidence["matched_rule"]["precision"])
    if action["source"] == "v36_selective_count_consensus":
        score = float(evidence.get("node_score", 0.5))
        return score if action["add_rids"] else 1 - score
    if action["source"] == "v40_boundary_frontier":
        rank = int(evidence.get("source_rank", 0))
        kind = "add" if action["add_rids"] else "delete"
        curve = read_json(V40)["oof_curves"][kind]
        for row in curve:
            if rank <= int(row["k"]):
                return float(row["precision"])
        return float(curve[-1]["precision"])
    # Rejected count-frontier rows have no validated probability calibration.
    return 0.5


def sidon_matrix(rows, columns, first_count, seed, attempts=50000):
    """Generate unique columns with no equal sums of two distinct columns."""
    rng = np.random.default_rng(seed)
    output, pair_sums = [], set()
    for index in range(columns):
        first = 1 if index < first_count else 0
        accepted = None
        for _ in range(attempts):
            candidate = np.r_[first, rng.binomial(1, 0.5, rows - 1)].astype(np.int8)
            if any(np.array_equal(candidate, old) for old in output):
                continue
            new_sums = [tuple((candidate + old).tolist()) for old in output]
            if len(new_sums) != len(set(new_sums)) or any(value in pair_sums for value in new_sums):
                continue
            accepted = candidate
            break
        if accepted is None:
            raise RuntimeError(f"failed to place coded column {index}")
        for old in output:
            pair_sums.add(tuple((accepted + old).tolist()))
        output.append(accepted)
    return np.stack(output, axis=1)


def alternative_exists(matrix, labels, time_limit=5.0):
    rhs = matrix @ labels
    ones = int(labels.sum())
    hamming = np.where(labels == 0, 1.0, -1.0)
    result = milp(
        -hamming, integrality=np.ones(len(labels), dtype=np.int8),
        bounds=Bounds(np.zeros(len(labels)), np.ones(len(labels))),
        constraints=LinearConstraint(matrix, rhs, rhs),
        options={"time_limit": time_limit},
    )
    if result.x is not None:
        distance = ones + float(hamming @ np.rint(result.x))
        if distance >= 0.5:
            # Any feasible non-identical assignment proves a collision even
            # when the optimizer stopped before proving global optimality.
            return True
        if result.success:
            return False
    return None


def recovery_diagnostic(matrix, probabilities, trials=10, seed=20260821):
    rng = np.random.default_rng(seed)
    unique = collisions = timeouts = 0
    for _ in range(trials):
        labels = (rng.random(len(probabilities)) < probabilities).astype(np.int8)
        result = alternative_exists(matrix, labels)
        if result is False:
            unique += 1
        elif result is True:
            collisions += 1
        else:
            timeouts += 1
    return {
        "trials": trials, "unique": unique, "collisions": collisions,
        "timeouts": timeouts, "proven_unique_rate": unique / trials,
        "warning": "Prior-sampled local uniqueness diagnostic; not a global injectivity proof.",
    }


def oracle_simulation(actions, probabilities, trials=50000, seed=20260822):
    rng = np.random.default_rng(seed)
    scores = []
    reached = 0
    for _ in range(trials):
        correct = rng.random(len(actions)) < probabilities
        add_correct = sum(bool(flag) and bool(action["add_rids"])
                          for flag, action in zip(correct, actions))
        deletes = sum(bool(action["remove_rids"]) for action in actions)
        delete_correct = sum(bool(flag) and bool(action["remove_rids"])
                             for flag, action in zip(correct, actions))
        # A selected deletion always lowers P.  It preserves TP only when the
        # removed node is a false positive; an incorrect deletion loses TP.
        delete_incorrect = deletes - delete_correct
        tp = BASE_TP + add_correct - delete_incorrect
        predictions = BASE_P + sum(bool(action["add_rids"]) for action in actions) - deletes
        score = 2 * tp / (TRUE_ROOTS + predictions)
        scores.append(score)
        reached += int(score >= TARGET)
    scores.sort()
    return {
        "trials": trials, "mean": sum(scores) / trials,
        "p05": scores[int(0.05 * trials)], "median": scores[trials // 2],
        "p95": scores[int(0.95 * trials)],
        "probability_reach_0_95": reached / trials,
        "warning": "Assumes every listed correctness bit is recovered and model probabilities are calibrated.",
    }


def emit_probe(row_index, row, actions, exact, order_ids, champion_roots, records_by_order):
    selected = [dict(action) for include, action in zip(row, actions) if include]
    all_actions = list(exact["actions"]) + selected
    roots = apply_actions(champion_roots, records_by_order, all_actions)
    additions = sum(len(action["add_rids"]) for action in selected)
    deletions = sum(len(action["remove_rids"]) for action in selected)
    predictions = sum(len(values) for values in roots.values())
    if predictions != BASE_P + additions - deletions:
        raise RuntimeError("coded prediction count mismatch")
    name = f"v58_code_{row_index + 1:02d}"
    path = OUT / "submissions" / f"{name}.csv"
    write_submission(path, order_ids, roots)
    manifest = {
        "probe_id": name, "row_index": row_index, "path": str(path),
        "baseline": str(CHAMPION), "fixed_base": exact["probe_id"],
        "candidate_indices": np.flatnonzero(row).astype(int).tolist(),
        "candidate_actions": selected, "actions": all_actions,
        "additions": additions, "deletions": deletions,
        "predictions": predictions, "prediction_delta": additions - deletions,
        "sha256": sha256(path),
        "score_possibilities": [{
            "correct_count": count, "batch_delta_tp": count - deletions,
            "tp": BASE_TP + count - deletions,
            "score": round(2 * (BASE_TP + count - deletions) /
                           (TRUE_ROOTS + predictions), 9),
        } for count in range(len(selected) + 1)],
    }
    (OUT / "manifests" / f"{name}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main():
    for directory in (OUT, OUT / "submissions", OUT / "manifests"):
        directory.mkdir(parents=True, exist_ok=True)
    actions, sources = collect_actions()
    probabilities = np.asarray([correctness_probability(action) for action in actions])
    matrix = sidon_matrix(ROWS, len(actions), sources["v49"], 20260820)
    order_ids, champion_roots = load_submission(CHAMPION)
    with gzip.open(RECORDS, "rt", encoding="utf-8") as handle:
        records = json.load(handle)["test"]
    records_by_order = {row["order_id"]: row for row in records}
    exact = read_json(V37 / "manifests/v37_exact_corrections.json")
    manifests = [
        emit_probe(index, row, actions, exact, order_ids, champion_roots, records_by_order)
        for index, row in enumerate(matrix)
    ]
    v49_sha = read_json(V49 / "report.json")["manifest"]["sha256"]
    if manifests[0]["sha256"] != v49_sha:
        raise RuntimeError("first coded row must exactly reproduce V49")
    report = {
        "version": "v58-coded-campaign-1", "rows": ROWS,
        "candidate_count": len(actions), "sources": sources,
        "base": {"tp": BASE_TP, "predictions": BASE_P,
                 "score": 2 * BASE_TP / (TRUE_ROOTS + BASE_P)},
        "target": TARGET, "estimated_days_at_two_submissions": (ROWS + 1 + 1) / 2,
        "matrix": matrix.astype(int).tolist(),
        "matrix_rank": int(np.linalg.matrix_rank(matrix)),
        "row_weights": matrix.sum(axis=1).astype(int).tolist(),
        "column_weights": matrix.sum(axis=0).astype(int).tolist(),
        "candidate_actions": actions,
        "correctness_probabilities": probabilities.tolist(),
        "recovery_diagnostic": recovery_diagnostic(matrix, probabilities),
        "oracle_diagnostic": oracle_simulation(actions, probabilities),
        "probe_order": [manifest["probe_id"] for manifest in manifests],
        "probes": {manifest["probe_id"]: manifest for manifest in manifests},
        "submission_plan": {
            "coded_probes": ROWS, "final_checkpoint": 1,
            "adaptive_reserve_if_not_unique": 1,
        },
    }
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    checkpoint_path = OUT / "highest_verified_checkpoint.csv"
    checkpoint_roots = apply_actions(champion_roots, records_by_order, exact["actions"])
    write_submission(checkpoint_path, order_ids, checkpoint_roots)
    checkpoint = {
        "path": str(checkpoint_path), "tp": BASE_TP, "predictions": BASE_P,
        "score": report["base"]["score"], "fixed_correct_count": 0,
        "fixed_incorrect_count": 0, "unresolved_count": len(actions),
        "sha256": sha256(checkpoint_path), "source": "v37_exact_base",
    }
    state = {
        "version": "v58-coded-state-1", "results": {},
        "checkpoint": checkpoint,
    }
    (OUT / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "highest_verified_checkpoint.json").write_text(
        json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "candidate_count": len(actions), "sources": sources,
        "estimated_days": report["estimated_days_at_two_submissions"],
        "recovery_diagnostic": report["recovery_diagnostic"],
        "oracle_diagnostic": report["oracle_diagnostic"],
        "first_probe": manifests[0]["path"], "first_sha256": manifests[0]["sha256"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
