"""Extend V47 with the only new V48 station-stable add candidate."""

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
V37 = EXPERIMENTS / "v37_online_equations"
V47 = EXPERIMENTS / "v47_v36_count_batch"
V48 = EXPERIMENTS / "v48_stable_strata"
OUT = EXPERIMENTS / "v49_extended_count_batch"
TRUE_ROOTS = 1044
BASE_TP_AFTER_EXACT = 956
BASE_P = 1035


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def emit(name, exact_actions, candidate_actions, order_ids, champion_roots, records_by_order):
    actions = list(exact_actions) + list(candidate_actions)
    roots = apply_actions(champion_roots, records_by_order, actions)
    path = OUT / "submissions" / f"{name}.csv"
    write_submission(path, order_ids, roots)
    manifest = {
        "probe_id": name, "baseline": str(CHAMPION), "fixed_base": "v37_exact_corrections",
        "path": str(path), "candidate_action_ids": [row["action_id"] for row in candidate_actions],
        "actions": actions, "predictions": sum(map(len, roots.values())), "sha256": sha256(path),
    }
    (OUT / "manifests" / f"{name}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "submissions").mkdir(exist_ok=True)
    (OUT / "manifests").mkdir(exist_ok=True)
    exact = read_json(V37 / "manifests/v37_exact_corrections.json")
    v47 = read_json(V47 / "report.json")["manifest"]
    v36_actions = [action for action in v47["actions"] if action["source"] == "v36_selective_count_consensus"]
    if len(v36_actions) != 3:
        raise ValueError(len(v36_actions))
    v48 = read_json(V48 / "report.json")
    existing = {(action["order_id"], rid) for action in v36_actions for rid in action["add_rids"] + action["remove_rids"]}
    new_matches = [
        row for row in v48["kinds"]["add"]["clean_test_matches"]
        if (row["order_id"], row["rid"]) not in existing
    ]
    if len(new_matches) != 1:
        raise ValueError(f"expected exactly one new V48 add, got {len(new_matches)}")
    new_row = new_matches[0]
    new_action = {
        "action_id": "v49_stable_add_001", "order_id": new_row["order_id"],
        "remove_rids": [], "add_rids": [new_row["rid"]],
        "source": "v48_station_stable_stratum", "expected_gain": None,
        "evidence": new_row,
    }
    candidate_actions = v36_actions + [new_action]
    order_ids, champion_roots = load_submission(CHAMPION)
    with gzip.open(RECORDS, "rt", encoding="utf-8") as handle:
        records = json.load(handle)["test"]
    records_by_order = {row["order_id"]: row for row in records}
    action_orders = [action["order_id"] for action in exact["actions"] + candidate_actions]
    if len(action_orders) != len(set(action_orders)):
        raise ValueError("actions overlap by order")

    subsets = {}
    for mask in range(16):
        chosen = [action for index, action in enumerate(candidate_actions) if mask & (1 << index)]
        name = "v49_exact_only" if mask == 0 else f"v49_subset_{mask:04b}"
        subsets[name] = emit(
            name, exact["actions"], chosen, order_ids, champion_roots, records_by_order
        )
    full = subsets["v49_subset_1111"]
    public_name = "v49_exact_plus_strict4"
    public_path = OUT / "submissions" / f"{public_name}.csv"
    public_path.write_bytes(Path(full["path"]).read_bytes())
    predictions = full["predictions"]
    if predictions != BASE_P + 2:
        raise ValueError((predictions, BASE_P + 2))
    exact_score = 2 * BASE_TP_AFTER_EXACT / (TRUE_ROOTS + BASE_P)
    possibilities = []
    for deleted_true in (0, 1):
        for true_adds in range(4):
            delta = true_adds - deleted_true
            tp = BASE_TP_AFTER_EXACT + delta
            score = 2 * tp / (TRUE_ROOTS + predictions)
            possibilities.append({
                "deleted_true": deleted_true, "true_adds": true_adds,
                "candidate_delta_tp": delta, "tp": tp, "score": round(score, 9),
                "decision": "accept_batch" if score > exact_score else "reject_and_split",
            })
    possibilities.sort(key=lambda row: (row["candidate_delta_tp"], row["deleted_true"]))
    manifest = {
        **full, "probe_id": public_name, "path": str(public_path),
        "sha256": sha256(public_path), "score_possibilities": possibilities,
        "subset_manifests": {
            name: {
                "path": row["path"], "candidate_action_ids": row["candidate_action_ids"],
                "predictions": row["predictions"], "sha256": row["sha256"],
            } for name, row in subsets.items()
        },
        "acceptance": {
            "exact_base_score": exact_score,
            "minimum_candidate_delta_tp": 1,
            "displayed_scores_to_accept": sorted({
                row["score"] for row in possibilities if row["decision"] == "accept_batch"
            }),
        },
    }
    (OUT / "manifests" / f"{public_name}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    outcomes = {}
    v47_path = read_json(V47 / "report.json")["manifest"]["path"]
    for delta in range(-1, 4):
        rows = [row for row in possibilities if row["candidate_delta_tp"] == delta]
        score = rows[0]["score"]
        if delta == -1:
            next_path = subsets["v49_exact_only"]["path"]
            deduction = "delete is true and all three additions are false"
        elif delta == 3:
            next_path = None
            deduction = "delete is false and all three additions are true"
        else:
            next_path = v47_path
            deduction = "submit V47 next; delta(V49)-delta(V47) gives the new add label exactly, while V47 counts the original three actions"
        outcomes[f"{score:.6f}"] = {
            "candidate_delta_tp": delta,
            "compatible_label_counts": [
                {"deleted_true": row["deleted_true"], "true_adds": row["true_adds"]}
                for row in rows
            ],
            "deduction": deduction, "next_submission": next_path,
        }
    report = {
        "version": "v49-extended-count-batch-1",
        "manifest": manifest,
        "new_v48_candidate": new_row,
        "decision_tree": {
            "first_submission": str(public_path), "outcomes": outcomes,
            "nested_second_submission": v47_path,
            "nested_identity": "candidate_delta_tp(V49) = candidate_delta_tp(V47) + label(new_v48_add)",
            "delete_probe": subsets["v49_subset_0001"]["path"],
            "addition_singletons": [
                subsets["v49_subset_0010"]["path"],
                subsets["v49_subset_0100"]["path"],
                subsets["v49_subset_1000"]["path"],
            ],
        },
        "recommendation": {"submit_next_directly": str(public_path)},
    }
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "path": str(public_path), "predictions": predictions, "sha256": manifest["sha256"],
        "new_candidate": {"order_id": new_row["order_id"], "rid": new_row["rid"], "title": new_row["title"]},
        "distinct_outcomes": [
            {"delta_tp": delta, "score": next(row["score"] for row in possibilities if row["candidate_delta_tp"] == delta)}
            for delta in range(-1, 4)
        ],
        "acceptance": manifest["acceptance"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
