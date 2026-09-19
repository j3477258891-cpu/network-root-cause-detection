"""Build selective per-order count correction probes from V34 consensus."""

from __future__ import annotations

import csv
import glob
import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\zgyidong")
if str(ROOT / "experiments/v30_meta_stack") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments/v30_meta_stack"))
from build_cross_order_probes import apply_actions, load_submission, write_submission


DATA = ROOT / "experiments/v25_semantic_router/cloud_dataset/v25_semantic_router.npz"
RECORDS = ROOT / "experiments/v25_semantic_router/cloud_dataset/semantic_records.json.gz"
CHAMPION = ROOT / "experiments/v30_meta_stack/submissions/v30_cross_order_top5.csv"
OUT = ROOT / "experiments/v36_selective_counts"
V11_REPORT = ROOT / "experiments/submissions/template_ranked/v11_constrained_report.json"
HISTORY_DIRS = (
    ROOT / "experiments/v27_closed_loop/manifests",
    ROOT / "experiments/v28_ten_day_campaign/manifests",
    ROOT / "experiments/v29_domain_ranker/manifests",
    ROOT / "experiments/v30_meta_stack/manifests",
)
TRUE_ROOTS = 1044
BASELINE_TP = 955
BASELINE_P = 1035
MAX_ROOTS = 8
DELETE_MEAN_THRESHOLD = 0.72
ADD_MEAN_THRESHOLD = 0.21
MIN_SUPPORT = 5


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def historical_nodes():
    nodes = set()
    for directory in HISTORY_DIRS:
        for path in directory.glob("*.json") if directory.exists() else []:
            try:
                data = read_json(path)
            except Exception:
                continue
            # Any prior action node is excluded. Even when the batch was not
            # individually resolved, reusing it would confound online deltas.
            for action in data.get("actions", []):
                oid = action.get("order_id")
                if not oid:
                    continue
                for rid in action.get("remove_rids", []) + action.get("add_rids", []):
                    nodes.add((oid, rid))
    return nodes


def protected_nodes():
    report = read_json(V11_REPORT)
    output = set()
    for key in ("forced_in", "forced_out"):
        output |= {(x["order_id"], x["rid"]) for x in report.get(key, [])}
    return output


def selected_mask(arrays, records, roots):
    ptr = arrays["test_alarm_ptr"]
    mask = np.zeros(len(arrays["test_v11"]), dtype=bool)
    for oi, (start, stop) in enumerate(zip(ptr[:-1], ptr[1:])):
        oid = records[oi]["order_id"]
        chosen = {node["@rid"] for node in roots[oid]}
        mask[int(start):int(stop)] = [a["rid"] in chosen for a in records[oi]["alarms"]]
    return mask


def score_possibilities(predictions, kind):
    deltas = (-1, 0) if kind == "delete" else (0, 1)
    return [
        {"delta_tp": d, "tp": BASELINE_TP + d, "predictions": predictions,
         "score": round(2 * (BASELINE_TP + d) / (TRUE_ROOTS + predictions), 9)}
        for d in deltas
    ]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "submissions").mkdir(exist_ok=True)
    (OUT / "manifests").mkdir(exist_ok=True)
    with np.load(DATA, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    with gzip.open(RECORDS, "rt", encoding="utf-8") as handle:
        records = json.load(handle)["test"]
    order_ids, champion_roots = load_submission(CHAMPION)
    records_by_order = {o["order_id"]: o for o in records}
    base = selected_mask(arrays, records, champion_roots)
    ptr = arrays["test_alarm_ptr"]
    node_score = np.load(ROOT / "experiments/v30_meta_stack/station_extra_trees_test.npy")
    probability_paths = sorted(glob.glob(str(ROOT / "experiments/v34_count_model/*_alpha_*_test.npy")))
    probabilities = [np.load(path) for path in probability_paths]
    historical = historical_nodes()
    protected = protected_nodes()

    candidates = []
    rejected = []
    for oi, (start, stop) in enumerate(zip(ptr[:-1], ptr[1:])):
        start, stop = int(start), int(stop)
        order = records[oi]
        oid = order["order_id"]
        current = int(base[start:stop].sum())
        limit = min(MAX_ROOTS, stop - start)
        predicted = [min(limit, int(np.argmax(p[oi]) + 1)) for p in probabilities]
        proposed, support = Counter(predicted).most_common(1)[0]
        if support < MIN_SUPPORT or proposed == current:
            continue
        direction = 1 if proposed > current else -1
        ranked = np.argsort(-node_score[start:stop], kind="stable")
        if direction > 0:
            local = next(int(j) for j in ranked if not base[start + j])
            kind = "add"
        else:
            local = next(int(j) for j in ranked[::-1] if base[start + j])
            kind = "delete"
        rid = order["alarms"][local]["rid"]
        margins = [float(p[oi, proposed - 1] - p[oi, current - 1]) for p in probabilities]
        mean_margin = float(np.mean(margins))
        threshold = ADD_MEAN_THRESHOLD if kind == "add" else DELETE_MEAN_THRESHOLD
        row = {
            "order_id": oid,
            "rid": rid,
            "kind": kind,
            "current_count": current,
            "proposed_count": proposed,
            "support": support,
            "mean_margin": mean_margin,
            "min_margin": min(margins),
            "node_score": float(node_score[start + local]),
            "probability_models": probability_paths,
        }
        if mean_margin < threshold:
            row["reason"] = "below_oof_frozen_threshold"
            rejected.append(row)
            continue
        if (oid, rid) in protected:
            row["reason"] = "v11_protected_node"
            rejected.append(row)
            continue
        if (oid, rid) in historical:
            row["reason"] = "historical_probe_node"
            rejected.append(row)
            continue
        candidates.append(row)

    candidates.sort(key=lambda row: (row["kind"] != "delete", -row["mean_margin"]))
    emitted = {}
    for index, candidate in enumerate(candidates, 1):
        name = f"v36_count_{candidate['kind']}_{index}"
        action = {
            "action_id": name,
            "order_id": candidate["order_id"],
            "remove_rids": [candidate["rid"]] if candidate["kind"] == "delete" else [],
            "add_rids": [candidate["rid"]] if candidate["kind"] == "add" else [],
            "source": "v36_selective_count_consensus",
            "expected_gain": candidate["mean_margin"],
            "evidence": candidate,
        }
        roots = apply_actions(champion_roots, records_by_order, [action])
        predictions = sum(len(nodes) for nodes in roots.values())
        path = OUT / "submissions" / f"{name}.csv"
        write_submission(path, order_ids, roots)
        manifest = {
            "probe_id": name,
            "baseline": str(CHAMPION),
            "path": str(path),
            "kind": candidate["kind"],
            "actions": [action],
            "predictions": predictions,
            "prediction_delta": predictions - BASELINE_P,
            "sha256": sha256(path),
            "score_possibilities": score_possibilities(predictions, candidate["kind"]),
        }
        (OUT / "manifests" / f"{name}.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        emitted[name] = {
            "path": str(path), "kind": candidate["kind"],
            "order_id": candidate["order_id"], "rid": candidate["rid"],
            "predictions": predictions, "sha256": manifest["sha256"],
        }

    report = {
        "version": "v36-selective-counts-1",
        "baseline": str(CHAMPION),
        "oof_frozen_thresholds": {
            "delete_mean_margin": DELETE_MEAN_THRESHOLD,
            "delete_curve": "top12/12 false positives; zero TP loss",
            "add_mean_margin": ADD_MEAN_THRESHOLD,
            "add_curve": "top15 added 11 TP; top20 added 14 TP",
            "minimum_model_support": MIN_SUPPORT,
        },
        "historical_nodes_excluded": len(historical),
        "protected_nodes": len(protected),
        "accepted_candidates": candidates,
        "rejected_candidates": rejected,
        "probes": emitted,
    }
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
