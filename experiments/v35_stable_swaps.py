"""Generate multi-model-stable within-order swap probes."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\zgyidong")
if str(ROOT / "experiments/v30_meta_stack") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments/v30_meta_stack"))
from build_cross_order_probes import apply_actions, load_submission, node_for_add, write_submission


DATA = ROOT / "experiments/v25_semantic_router/cloud_dataset/v25_semantic_router.npz"
RECORDS = ROOT / "experiments/v25_semantic_router/cloud_dataset/semantic_records.json.gz"
CHAMPION = ROOT / "experiments/v30_meta_stack/submissions/v30_cross_order_top5.csv"
OUT = ROOT / "experiments/v35_stable_swaps"
V30_REPORT = ROOT / "experiments/v30_meta_stack/v30_cross_order_report.json"
TEMPLATE_REPORT = ROOT / "experiments/submissions/template_ranked/v11_constrained_report.json"
HISTORY_DIRS = (
    ROOT / "experiments/v28_ten_day_campaign/manifests",
    ROOT / "experiments/v29_domain_ranker/manifests",
    ROOT / "experiments/v30_meta_stack/manifests",
)
TRUE_ROOTS = 1044
BASELINE_TP = 955
BASELINE_P = 1035
MAX_ROOTS = 8


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_mask(arrays, records, roots):
    ptr = arrays["test_alarm_ptr"]
    mask = np.zeros(len(arrays["test_v11"]), dtype=bool)
    for oi, (start, stop) in enumerate(zip(ptr[:-1], ptr[1:])):
        oid = records["test"][oi]["order_id"]
        selected = {n["@rid"] for n in roots[oid]}
        for j, alarm in enumerate(records["test"][oi]["alarms"]):
            mask[int(start) + j] = alarm["rid"] in selected
    return mask


def exclusions():
    touched = set()
    protected = set()
    for directory in HISTORY_DIRS:
        for path in directory.glob("*.json") if directory.exists() else []:
            try:
                data = read_json(path)
            except Exception:
                continue
            for action in data.get("actions", []):
                if action.get("order_id"):
                    touched.add(action["order_id"])
                oid = action.get("order_id")
                # Historical actions identify touched orders, but their
                # individual nodes are not protected unless V11 forced them.
                # Aggregate leaderboard probes cannot establish node-level
                # truth, so treating every touched RID as protected would
                # discard all remaining signal.
    if TEMPLATE_REPORT.exists():
        report = read_json(TEMPLATE_REPORT)
        for key in ("forced_in", "forced_out"):
            protected |= {(x["order_id"], x["rid"]) for x in report.get(key, [])}
    if V30_REPORT.exists():
        report = read_json(V30_REPORT)
        for action in report.get("probes", {}).get("v30_cross_order_top5", {}).get("actions", []):
            touched.add(action["order_id"])
    return touched, protected


def historical_nodes():
    """Nodes used by scored probes are not safe for a new swap."""
    nodes = set()
    for directory in HISTORY_DIRS:
        for path in directory.glob("*.json") if directory.exists() else []:
            try:
                data = read_json(path)
            except Exception:
                continue
            for action in data.get("actions", []):
                oid = action.get("order_id")
                if oid:
                    nodes |= {(oid, rid) for rid in action.get("remove_rids", [])}
                    nodes |= {(oid, rid) for rid in action.get("add_rids", [])}
    return nodes


def stable_action_rows(scores, base, ptr, records, touched, protected,
                       historical, allow_touched=False):
    """Keep only exact same one-order replacement across all score models."""
    order_count = len(ptr) - 1
    proposals = []
    for oi, (start, stop) in enumerate(zip(ptr[:-1], ptr[1:])):
        start, stop = int(start), int(stop)
        oid = records["test"][oi]["order_id"]
        if oid in touched and not allow_touched:
            continue
        count = int(base[start:stop].sum())
        selected_sets = []
        for score in scores:
            ranked = np.argsort(-score[start:stop], kind="stable")[:count]
            proposed = np.zeros(stop - start, dtype=bool)
            proposed[ranked] = True
            selected_sets.append(proposed)
        groups = []
        for candidate in selected_sets:
            for group in groups:
                if np.array_equal(group[0], candidate):
                    group.append(candidate)
                    break
            else:
                groups.append([candidate])
        group = max(groups, key=len)
        if len(group) < 2:
            continue
        proposed = group[0]
        added = np.flatnonzero(proposed & ~base[start:stop])
        removed = np.flatnonzero(base[start:stop] & ~proposed)
        if len(added) == 0 or len(removed) == 0 or len(added) != len(removed):
            continue
        if len(added) > 2 or count - len(removed) < 1:
            continue
        alarms = records["test"][oi]["alarms"]
        add_rids = [alarms[int(j)]["rid"] for j in added]
        remove_rids = [alarms[int(j)]["rid"] for j in removed]
        if any((oid, rid) in protected | historical for rid in add_rids + remove_rids):
            continue
        utilities = []
        supporting = []
        for model_index, score in enumerate(scores):
            if not np.array_equal(selected_sets[model_index], proposed):
                continue
            supporting.append(model_index)
            utilities.append(float(score[start:stop][added].sum() - score[start:stop][removed].sum()))
        proposals.append({
            "order_index": oi,
            "order_id": oid,
            "remove_rids": remove_rids,
            "add_rids": add_rids,
            "model_utility_min": min(utilities),
            "model_utility_mean": float(np.mean(utilities)),
            "model_utility_values": utilities,
            "supporting_model_indices": supporting,
            "count": count,
        })
    proposals.sort(key=lambda row: (row["model_utility_min"], row["model_utility_mean"]), reverse=True)
    return proposals


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "submissions").mkdir(exist_ok=True)
    (OUT / "manifests").mkdir(exist_ok=True)
    with np.load(DATA, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    with gzip.open(RECORDS, "rt", encoding="utf-8") as handle:
        records = json.load(handle)
    order_ids, champion_roots = load_submission(CHAMPION)
    records_by_order = {o["order_id"]: o for o in records["test"]}
    base = test_mask(arrays, records, champion_roots)
    ptr = arrays["test_alarm_ptr"]
    test_scores = [
        np.load(ROOT / "experiments/v30_meta_stack/station_extra_trees_test.npy"),
        np.load(OUT.parent / "v33_domain_adaptation/station_alpha_0.5_test.npy"),
        np.load(OUT.parent / "v33_domain_adaptation/station_alpha_1_test.npy"),
        np.load(OUT.parent / "v33_domain_adaptation/station_alpha_2_test.npy"),
    ]
    touched, protected = exclusions()
    historical = historical_nodes()
    candidates = stable_action_rows(
        test_scores, base, ptr, records, touched, protected, historical,
        allow_touched=True
    )
    if len(candidates) < 3:
        report = {
            "version": "v35-stable-swaps-1",
            "baseline": str(CHAMPION),
            "candidate_count": len(candidates),
            "excluded_orders": len(touched),
            "protected_nodes": len(protected),
            "historical_nodes_excluded": len(historical),
            "top_candidates": candidates,
            "probes": {},
            "offline_gate": {
                "passed": False,
                "reason": "all remaining stable changes reuse RIDs from scored probes",
            },
        }
        (OUT / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    batches = {
        "v35_stable_single1": 1,
        "v35_stable_single2": 1,
        "v35_stable_single3": 1,
    }
    summary = {}
    for batch_index, (name, count) in enumerate(batches.items()):
        selected = candidates[batch_index:batch_index + count]
        actions = []
        for i, item in enumerate(selected, 1):
            actions.append({
                "action_id": f"v35_stable_{i:03d}_{item['order_id'][:8]}",
                "order_id": item["order_id"],
                "remove_rids": item["remove_rids"],
                "add_rids": item["add_rids"],
                "source": "v35_multi_model_stable_swap",
                "expected_gain": item["model_utility_min"],
                "evidence": item,
            })
        roots = apply_actions(champion_roots, records_by_order, actions)
        predictions = sum(len(nodes) for nodes in roots.values())
        if predictions != BASELINE_P:
            raise ValueError((name, predictions))
        path = OUT / "submissions" / f"{name}.csv"
        write_submission(path, order_ids, roots)
        manifest = {
            "probe_id": name,
            "baseline": str(CHAMPION),
            "path": str(path),
            "kind": "swap",
            "actions": actions,
            "predictions": predictions,
            "prediction_delta": 0,
            "sha256": sha256(path),
            "excluded_orders": len(touched),
            "protected_nodes": len(protected),
            "score_possibilities": [
                {"delta_tp": d, "tp": BASELINE_TP + d,
                 "score": round(2 * (BASELINE_TP + d) / (TRUE_ROOTS + BASELINE_P), 9)}
                for d in range(-count, count + 1)
            ],
        }
        (OUT / "manifests" / f"{name}.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary[name] = {
            "predictions": predictions,
            "sha256": manifest["sha256"],
            "orders": [x["order_id"] for x in selected],
        }
    report = {
        "version": "v35-stable-swaps-1",
        "baseline": str(CHAMPION),
        "candidate_count": len(candidates),
        "excluded_orders": len(touched),
        "protected_nodes": len(protected),
        "historical_nodes_excluded": len(historical),
        "top_candidates": candidates[:20],
        "probes": summary,
        "offline_gate": {
            "stable_oof_top5_delta_tp": 2,
            "stable_oof_top10_delta_tp": 3,
            "stable_oof_top20_delta_tp": 7,
            "note": "These are OOF estimates, not leaderboard guarantees.",
            "historical_node_gate": "all candidate RIDs already appeared in scored probes; no safe action remains"
        }
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
