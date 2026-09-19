"""Combine all three strict V36 count actions into one leaderboard count batch."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V30 = ROOT / "experiments/v30_meta_stack"
if str(V30) not in sys.path:
    sys.path.insert(0, str(V30))

from build_cross_order_probes import apply_actions, load_submission, write_submission


EXPERIMENTS = ROOT / "experiments"
CHAMPION = V30 / "submissions/v30_cross_order_top5.csv"
RECORDS = EXPERIMENTS / "v25_semantic_router/cloud_dataset/semantic_records.json.gz"
V36 = EXPERIMENTS / "v36_selective_counts"
V37 = EXPERIMENTS / "v37_online_equations"
OUT = EXPERIMENTS / "v47_v36_count_batch"
TRUE_ROOTS = 1044
BASE_TP_AFTER_EXACT = 956
BASE_P_AFTER_EXACT = 1035


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def emit(name, exact_actions, candidate_actions, order_ids, champion_roots, records_by_order):
    actions = list(exact_actions) + list(candidate_actions)
    roots = apply_actions(champion_roots, records_by_order, actions)
    predictions = sum(map(len, roots.values()))
    path = OUT / "submissions" / f"{name}.csv"
    write_submission(path, order_ids, roots)
    manifest = {
        "probe_id": name, "baseline": str(CHAMPION),
        "fixed_base": "v37_exact_corrections", "path": str(path),
        "candidate_action_ids": [action["action_id"] for action in candidate_actions],
        "actions": actions, "predictions": predictions, "sha256": sha256(path),
    }
    (OUT / "manifests" / f"{name}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "submissions").mkdir(exist_ok=True)
    (OUT / "manifests").mkdir(exist_ok=True)
    order_ids, champion_roots = load_submission(CHAMPION)
    with gzip.open(RECORDS, "rt", encoding="utf-8") as handle:
        records = json.load(handle)["test"]
    records_by_order = {row["order_id"]: row for row in records}
    exact = read_json(V37 / "manifests/v37_exact_corrections.json")
    v36_report = read_json(V36 / "report.json")
    probe_names = ["v36_count_delete_1", "v36_count_add_2", "v36_count_add_3"]
    candidate_actions = []
    source_manifests = []
    for name in probe_names:
        manifest = read_json(V36 / "manifests" / f"{name}.json")
        source_manifests.append({
            "probe_id": name, "path": manifest["path"], "sha256": manifest["sha256"]
        })
        candidate_actions.extend(manifest["actions"])
    actions = list(exact["actions"]) + candidate_actions
    action_orders = [action["order_id"] for action in actions]
    if len(action_orders) != len(set(action_orders)):
        raise ValueError("V37 and V36 actions are not order-disjoint")
    subsets = {}
    for mask in range(8):
        chosen = [action for index, action in enumerate(candidate_actions) if mask & (1 << index)]
        subset_name = "v47_exact_only" if mask == 0 else f"v47_subset_{mask:03b}"
        subsets[subset_name] = emit(
            subset_name, exact["actions"], chosen, order_ids, champion_roots, records_by_order
        )
    full = subsets["v47_subset_111"]
    predictions = full["predictions"]
    if predictions != BASE_P_AFTER_EXACT + 1:
        raise ValueError((predictions, BASE_P_AFTER_EXACT + 1))

    name = "v47_exact_plus_v36_all3"
    path = OUT / "submissions" / f"{name}.csv"
    # Keep the descriptive public name byte-identical to subset_111.
    roots = apply_actions(champion_roots, records_by_order, actions)
    write_submission(path, order_ids, roots)
    exact_score = 2 * BASE_TP_AFTER_EXACT / (TRUE_ROOTS + BASE_P_AFTER_EXACT)
    possibilities = []
    for deleted_true in (0, 1):
        for true_adds in (0, 1, 2):
            delta = true_adds - deleted_true
            tp = BASE_TP_AFTER_EXACT + delta
            score = 2 * tp / (TRUE_ROOTS + predictions)
            possibilities.append({
                "deleted_true": deleted_true,
                "true_adds": true_adds,
                "candidate_delta_tp": delta,
                "tp": tp,
                "score": round(score, 9),
                "decision": "accept_batch" if score > exact_score else "reject_and_split",
            })
    possibilities.sort(key=lambda row: (row["candidate_delta_tp"], row["deleted_true"]))
    manifest = {
        "probe_id": name,
        "baseline": str(CHAMPION),
        "fixed_base": exact["probe_id"],
        "path": str(path),
        "source_manifests": source_manifests,
        "actions": actions,
        "candidate_action_count": len(candidate_actions),
        "predictions": predictions,
        "prediction_delta_vs_exact": predictions - BASE_P_AFTER_EXACT,
        "sha256": sha256(path),
        "score_possibilities": possibilities,
        "acceptance": {
            "exact_base_score": exact_score,
            "minimum_candidate_delta_tp": 1,
            "displayed_scores_to_accept": sorted({
                row["score"] for row in possibilities if row["decision"] == "accept_batch"
            }),
        },
        "heuristic_only": {
            "delete_oof_curve": v36_report["oof_frozen_thresholds"]["delete_curve"],
            "add_oof_curve": v36_report["oof_frozen_thresholds"]["add_curve"],
            "warning": "OOF rates are not guarantees and are not used for the online accept decision.",
        },
        "subset_manifests": {
            subset_name: {
                "path": row["path"], "candidate_action_ids": row["candidate_action_ids"],
                "predictions": row["predictions"], "sha256": row["sha256"],
            }
            for subset_name, row in subsets.items()
        },
    }
    (OUT / "manifests" / f"{name}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    exact_only = subsets["v47_exact_only"]
    delete_only = subsets["v47_subset_001"]
    add2_only = subsets["v47_subset_010"]
    add3_only = subsets["v47_subset_100"]
    decision_tree = {
        "first_submission": str(path),
        "why_first": "It contains the proof-positive V37 correction, so it dominates submitting V37 alone as an information-gathering step.",
        "outcomes": {
            "0.918269": {
                "candidate_delta_tp": -1,
                "deduction": "delete node is true; both additions are false",
                "next_submission": exact_only["path"],
                "final_action_subset": [],
            },
            "0.919231": {
                "candidate_delta_tp": 0,
                "deduction": "either all candidate labels are false, or delete is true and exactly one addition is true",
                "next_submission": delete_only["path"],
                "next_probe_purpose": "resolve the delete label; if true, probe one addition next",
            },
            "0.920192": {
                "candidate_delta_tp": 1,
                "deduction": "either delete is false with exactly one true addition, or delete and both additions are true",
                "next_submission": delete_only["path"],
                "next_probe_purpose": "resolve the delete label; this also determines whether one or two additions are true",
            },
            "0.921154": {
                "candidate_delta_tp": 2,
                "deduction": "delete is false; both additions are true",
                "next_submission": None,
                "final_action_subset": [action["action_id"] for action in candidate_actions],
            },
        },
        "delete_probe_outcomes": {
            "0.919153": {
                "delete_label": 1,
                "meaning": "deletion is harmful; retain the node",
                "addition_probe_if_needed": add2_only["path"],
            },
            "0.920115": {
                "delete_label": 0,
                "meaning": "deletion is beneficial; retain the deletion",
                "addition_probe_if_needed": add2_only["path"],
            },
        },
        "add_probe_paths": [add2_only["path"], add3_only["path"]],
    }
    report = {
        "version": "v47-v36-count-batch-1",
        "manifest": manifest,
        "decision_tree": decision_tree,
        "recommendation": {
            "submit_next_directly": str(path),
            "accept_only_if_candidate_delta_tp_at_least": 1,
        },
    }
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "path": str(path), "predictions": predictions, "sha256": manifest["sha256"],
        "score_possibilities": possibilities,
        "acceptance": manifest["acceptance"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
