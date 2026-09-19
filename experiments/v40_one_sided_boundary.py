"""Audit one-sided add and delete batches independently of fixed-count swaps."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(r"D:\zgyidong")
if str(ROOT / ".deps") not in sys.path:
    sys.path.insert(0, str(ROOT / ".deps"))
if str(ROOT / "experiments/v30_meta_stack") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments/v30_meta_stack"))

import numpy as np

from build_cross_order_probes import apply_actions, load_submission, write_submission
from v30_meta_stack import exact_count_mask


DATA = ROOT / "experiments/v25_semantic_router/cloud_dataset/v25_semantic_router.npz"
RECORDS = ROOT / "experiments/v25_semantic_router/cloud_dataset/semantic_records.json.gz"
V30 = ROOT / "experiments/v30_meta_stack"
V33 = ROOT / "experiments/v33_domain_adaptation"
V37 = ROOT / "experiments/v37_online_equations"
CHAMPION = V30 / "submissions/v30_cross_order_top5.csv"
OUT = ROOT / "experiments/v40_one_sided_boundary"
TRAIN_BUDGET = 3103
BASE_TP = 955
BASE_P = 1035
TRUE_ROOTS = 1044
MAX_ROOTS = 8


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_mask(records, ptr, roots):
    result = np.zeros(int(ptr[-1]), dtype=bool)
    for order, start, stop in zip(records, ptr[:-1], ptr[1:]):
        chosen = {node["@rid"] for node in roots[order["order_id"]]}
        result[int(start):int(stop)] = [alarm["rid"] in chosen for alarm in order["alarms"]]
    return result


def score_arrays(arrays, split):
    suffix = "oof" if split == "train" else "test"
    return {
        "v11": arrays[f"{split}_v11"].astype(np.float32),
        "v13": arrays[f"{split}_v13"].astype(np.float32),
        "v19": arrays[f"{split}_v19"].astype(np.float32),
        "station": np.load(V30 / f"station_extra_trees_{suffix}.npy"),
        "domain": np.load(V33 / f"station_alpha_2_{suffix}.npy"),
        "consensus": np.load(V30 / f"v30_consensus_{suffix}.npy"),
    }


def rank_average(scores):
    values = []
    for score in scores.values():
        order = np.argsort(score, kind="stable")
        rank = np.empty(len(score), dtype=np.float32)
        rank[order] = np.linspace(0, 1, len(score), dtype=np.float32)
        values.append(rank)
    return np.mean(values, axis=0)


def unique_boundary(mask, ptr, score, kind, maximum=200):
    counts = np.asarray([mask[int(s):int(t)].sum() for s, t in zip(ptr[:-1], ptr[1:])])
    order_for_row = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
    if kind == "add":
        valid = (~mask) & (counts[order_for_row] < MAX_ROOTS)
        ranked = np.argsort(-score, kind="stable")
    else:
        valid = mask & (counts[order_for_row] > 1)
        ranked = np.argsort(score, kind="stable")
    output, used = [], set()
    for row in ranked:
        oi = int(order_for_row[row])
        if not valid[row] or oi in used:
            continue
        used.add(oi)
        output.append(int(row))
        if len(output) >= maximum:
            break
    return output, order_for_row


def curve(rows, labels, folds, kind):
    output = []
    target = labels if kind == "add" else 1 - labels
    for k in (5, 8, 10, 16, 20, 24, 32, 40, 60, 80, 100, 120, 150, 200):
        chosen = np.asarray(rows[:min(k, len(rows))], dtype=np.int64)
        if not len(chosen):
            continue
        fold_values = []
        for fold in range(5):
            local = chosen[folds[chosen] == fold]
            if len(local):
                fold_values.append(float(target[local].mean()))
        output.append({
            "k": len(chosen), "correct": int(target[chosen].sum()),
            "precision": float(target[chosen].mean()),
            "minimum_fold_precision": min(fold_values) if fold_values else None,
        })
    return output


def candidate_rows(rows, records, ptr, scores, average, fixed, exact_orders, kind):
    order_for_row = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
    local_for_row = np.concatenate([np.arange(int(t - s)) for s, t in zip(ptr[:-1], ptr[1:])])
    output = []
    for row in rows:
        oi = int(order_for_row[row])
        order = records[oi]
        alarm = order["alarms"][int(local_for_row[row])]
        key = (order["order_id"], alarm["rid"])
        if key in fixed or order["order_id"] in exact_orders:
            continue
        output.append({
            "order_id": key[0], "rid": key[1], "kind": kind,
            "title": alarm.get("title", ""), "rank_average": float(average[row]),
            "scores": {name: float(value[row]) for name, value in scores.items()},
        })
    return output


def emit(name, candidates, exact_actions, order_ids, champion_roots, records_by_order):
    actions = [dict(action) for action in exact_actions]
    for index, row in enumerate(candidates, 1):
        actions.append({
            "action_id": f"{name}_{index:03d}", "order_id": row["order_id"],
            "remove_rids": [row["rid"]] if row["kind"] == "delete" else [],
            "add_rids": [row["rid"]] if row["kind"] == "add" else [],
            "source": "v40_one_sided_rank_average", "expected_gain": None,
            "evidence": row,
        })
    roots = apply_actions(champion_roots, records_by_order, actions)
    predictions = sum(map(len, roots.values()))
    path = OUT / "submissions" / f"{name}.csv"
    write_submission(path, order_ids, roots)
    possibilities = []
    if candidates[0]["kind"] == "add":
        for correct in range(len(candidates) + 1):
            tp = BASE_TP + 1 + correct
            possibilities.append({"true_adds": correct, "tp": tp,
                                  "score": round(2 * tp / (TRUE_ROOTS + predictions), 9)})
    else:
        for lost in range(len(candidates) + 1):
            tp = BASE_TP + 1 - lost
            possibilities.append({"deleted_true": lost, "tp": tp,
                                  "score": round(2 * tp / (TRUE_ROOTS + predictions), 9)})
    manifest = {
        "probe_id": name, "baseline": str(CHAMPION), "path": str(path),
        "known_exact_gain": 1, "candidate_count": len(candidates), "actions": actions,
        "predictions": predictions, "sha256": sha256(path),
        "score_possibilities": possibilities,
    }
    (OUT / "manifests" / f"{name}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "submissions").mkdir(exist_ok=True)
    (OUT / "manifests").mkdir(exist_ok=True)
    with np.load(DATA, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    with gzip.open(RECORDS, "rt", encoding="utf-8") as handle:
        records = json.load(handle)
    train_scores, test_scores = score_arrays(arrays, "train"), score_arrays(arrays, "test")
    train_average, test_average = rank_average(train_scores), rank_average(test_scores)
    train_base = exact_count_mask(train_average, arrays["train_alarm_ptr"], TRAIN_BUDGET)
    order_ids, champion_roots = load_submission(CHAMPION)
    test_base = selected_mask(records["test"], arrays["test_alarm_ptr"], champion_roots)
    folds_by_order = arrays["train_station_folds"].astype(np.int8)
    train_order_for_row = np.repeat(np.arange(len(folds_by_order)), np.diff(arrays["train_alarm_ptr"]))
    folds = folds_by_order[train_order_for_row]
    fixed_report = read_json(V37 / "report.json")
    fixed = {(row["order_id"], row["rid"]) for row in fixed_report["fixed_labels"]}
    exact = read_json(V37 / "manifests/v37_exact_corrections.json")
    exact_actions = exact["actions"]
    exact_orders = {action["order_id"] for action in exact_actions}
    records_by_order = {row["order_id"]: row for row in records["test"]}
    report_curves, candidates, probes = {}, {}, {}
    for kind in ("add", "delete"):
        train_rows, _ = unique_boundary(
            train_base, arrays["train_alarm_ptr"], train_average, kind
        )
        test_rows, _ = unique_boundary(
            test_base, arrays["test_alarm_ptr"], test_average, kind
        )
        report_curves[kind] = curve(
            train_rows, arrays["train_labels"].astype(np.int8), folds, kind
        )
        candidates[kind] = candidate_rows(
            test_rows, records["test"], arrays["test_alarm_ptr"], test_scores,
            test_average, fixed, exact_orders, kind,
        )
        for size in (8, 16, 24, 32):
            subset = candidates[kind][:size]
            if not subset:
                continue
            name = f"v40_{kind}_top{len(subset):02d}_plus_exact"
            probes[name] = emit(
                name, subset, exact_actions, order_ids, champion_roots, records_by_order
            )
            if len(subset) < size:
                break
    report = {
        "version": "v40-one-sided-boundary-1", "oof_curves": report_curves,
        "test_candidates": candidates, "probes": probes,
        "f1_acceptance_threshold": BASE_TP / (TRUE_ROOTS + BASE_P),
    }
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "oof_curves": report_curves,
        "test_candidate_counts": {kind: len(rows) for kind, rows in candidates.items()},
        "top_candidates": {kind: rows[:10] for kind, rows in candidates.items()},
        "probes": {name: {k: value[k] for k in ("path", "predictions", "sha256")}
                   for name, value in probes.items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
