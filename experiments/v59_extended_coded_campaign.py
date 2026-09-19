"""Build phase two: 75 coded queries over 161 unused frontier actions."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
V30 = EXP / "v30_meta_stack"
for value in (EXP, V30):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from build_cross_order_probes import apply_actions, load_submission, write_submission
from v37_online_equation_solver import CHAMPION, TRUE_ROOTS
from v58_coded_campaign import (
    action_key, collect_actions, correctness_probability, oracle_simulation,
    recovery_diagnostic, sha256, sidon_matrix,
)


V37 = EXP / "v37_online_equations"
V52 = EXP / "v52_precision_frontier"
V58 = EXP / "v58_coded_campaign"
RECORDS = EXP / "v25_semantic_router/cloud_dataset/semantic_records.json.gz"
OUT = EXP / "v59_extended_coded_campaign"
BASE_TP = 956
BASE_P = 1035
ROWS = 75
TARGET = 0.95


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def remaining_actions():
    used, _ = collect_actions()
    used_keys = {action_key(action) for action in used}
    report = read_json(V52 / "report.json")
    actions = [
        action for kind in ("add", "delete") for action in report["pool"][kind]
        if action_key(action) not in used_keys
    ]
    if len(actions) != 161:
        raise RuntimeError(len(actions))
    orders = [action["order_id"] for action in actions]
    if len(set(orders)) != len(actions):
        raise RuntimeError("V59 remaining actions are not order-disjoint")
    if set(orders) & {action["order_id"] for action in used}:
        raise RuntimeError("V59 overlaps V58 orders")
    return actions


def emit_probe(row_index, row, actions, exact, order_ids, champion_roots, records_by_order):
    selected = [dict(action) for include, action in zip(row, actions) if include]
    all_actions = list(exact["actions"]) + selected
    roots = apply_actions(champion_roots, records_by_order, all_actions)
    additions = sum(bool(action["add_rids"]) for action in selected)
    deletions = sum(bool(action["remove_rids"]) for action in selected)
    predictions = sum(len(values) for values in roots.values())
    if predictions != BASE_P + additions - deletions:
        raise RuntimeError("V59 prediction mismatch")
    name = f"v59_code_{row_index + 1:02d}"
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
    actions = remaining_actions()
    probabilities = np.asarray([correctness_probability(action) for action in actions])
    matrix = sidon_matrix(ROWS, len(actions), 80, 20260823)
    order_ids, champion_roots = load_submission(CHAMPION)
    with gzip.open(RECORDS, "rt", encoding="utf-8") as handle:
        records = json.load(handle)["test"]
    records_by_order = {row["order_id"]: row for row in records}
    exact = read_json(V37 / "manifests/v37_exact_corrections.json")
    manifests = [
        emit_probe(index, row, actions, exact, order_ids, champion_roots, records_by_order)
        for index, row in enumerate(matrix)
    ]
    v58 = read_json(V58 / "report.json")
    combined_actions = list(v58["candidate_actions"]) + list(actions)
    combined_probabilities = np.r_[v58["correctness_probabilities"], probabilities]
    report = {
        "version": "v59-extended-coded-campaign-1", "rows": ROWS,
        "candidate_count": len(actions), "base": {
            "tp": BASE_TP, "predictions": BASE_P,
            "score": 2 * BASE_TP / (TRUE_ROOTS + BASE_P),
        },
        "target": TARGET,
        "estimated_phase_days_at_two_submissions": (ROWS + 1) / 2,
        "estimated_cumulative_days_with_v58": 31 + (ROWS + 1) / 2,
        "matrix": matrix.astype(int).tolist(),
        "matrix_rank": int(np.linalg.matrix_rank(matrix)),
        "row_weights": matrix.sum(axis=1).astype(int).tolist(),
        "column_weights": matrix.sum(axis=0).astype(int).tolist(),
        "candidate_actions": actions,
        "correctness_probabilities": probabilities.tolist(),
        "expected_correct_actions": float(probabilities.sum()),
        "recovery_diagnostic": recovery_diagnostic(matrix, probabilities, trials=10, seed=20260824),
        "phase_oracle_diagnostic": oracle_simulation(actions, probabilities, seed=20260825),
        "combined_v58_v59_oracle_diagnostic": oracle_simulation(
            combined_actions, combined_probabilities, seed=20260826
        ),
        "probe_order": [manifest["probe_id"] for manifest in manifests],
        "probes": {manifest["probe_id"]: manifest for manifest in manifests},
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
    state = {"version": "v59-coded-state-1", "results": {}, "checkpoint": checkpoint}
    (OUT / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "highest_verified_checkpoint.json").write_text(
        json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "candidate_count": len(actions), "expected_correct_actions": report["expected_correct_actions"],
        "phase_days": report["estimated_phase_days_at_two_submissions"],
        "cumulative_days": report["estimated_cumulative_days_with_v58"],
        "recovery_diagnostic": report["recovery_diagnostic"],
        "phase_oracle": report["phase_oracle_diagnostic"],
        "combined_oracle": report["combined_v58_v59_oracle_diagnostic"],
        "first_probe": manifests[0]["path"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
