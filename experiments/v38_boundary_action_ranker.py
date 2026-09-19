"""Rank selective add/delete corrections at the current count boundary."""

from __future__ import annotations

import csv
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
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

from build_cross_order_probes import apply_actions, load_submission, write_submission
from v30_meta_stack import exact_count_mask


DATA = ROOT / "experiments/v25_semantic_router/cloud_dataset/v25_semantic_router.npz"
RECORDS = ROOT / "experiments/v25_semantic_router/cloud_dataset/semantic_records.json.gz"
V30 = ROOT / "experiments/v30_meta_stack"
V33 = ROOT / "experiments/v33_domain_adaptation"
V34 = ROOT / "experiments/v34_count_model"
V37 = ROOT / "experiments/v37_online_equations"
CHAMPION = V30 / "submissions/v30_cross_order_top5.csv"
EXACT_MANIFEST = V37 / "manifests/v37_exact_corrections.json"
OUT = ROOT / "experiments/v38_boundary_actions"
SEED = 20260820
TRAIN_BUDGET = 3103
TRUE_ROOTS = 1044
BASE_TP = 955
BASE_P = 1035
MAX_ROOTS = 8
MAX_STEPS = 3


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_mask(records, ptr, roots):
    mask = np.zeros(int(ptr[-1]), dtype=bool)
    for order, start, stop in zip(records, ptr[:-1], ptr[1:]):
        chosen = {node["@rid"] for node in roots[order["order_id"]]}
        mask[int(start):int(stop)] = [alarm["rid"] in chosen for alarm in order["alarms"]]
    return mask


def probability_paths(split: str):
    return [
        V34 / f"station_alpha_{alpha}_{split}.npy" for alpha in (0, 1, 2)
    ] + [
        V34 / f"template_alpha_{alpha}_{split}.npy" for alpha in (0, 1, 2)
    ]


def node_score_arrays(arrays, split: str):
    suffix = "oof" if split == "train" else "test"
    return [
        arrays[f"{split}_v11"], arrays[f"{split}_v13"], arrays[f"{split}_v19"],
        np.load(V30 / f"station_extra_trees_{suffix}.npy"),
        np.load(V33 / f"station_alpha_2_{suffix}.npy"),
        np.load(V30 / f"v30_consensus_{suffix}.npy"),
    ]


def make_actions(arrays, split, base, count_probabilities, node_scores, labels=None):
    ptr = arrays[f"{split}_alarm_ptr"]
    alarm_x = arrays[f"{split}_alarm_x"]
    rows, meta, targets = [], [], []
    for oi, (start, stop) in enumerate(zip(ptr[:-1], ptr[1:])):
        start, stop = int(start), int(stop)
        current = int(base[start:stop].sum())
        limit = min(MAX_ROOTS, stop - start)
        if current < 1 or current > limit:
            raise ValueError((split, oi, current, limit))
        score_matrix = np.column_stack([score[start:stop] for score in node_scores])
        consensus = score_matrix[:, -1]
        selected_local = np.flatnonzero(base[start:stop])
        unselected_local = np.flatnonzero(~base[start:stop])
        selected_ranked = selected_local[np.argsort(consensus[selected_local], kind="stable")]
        unselected_ranked = unselected_local[np.argsort(-consensus[unselected_local], kind="stable")]
        order_summary = np.concatenate([
            score_matrix.mean(0), score_matrix.std(0), score_matrix.max(0),
            score_matrix.min(0), np.asarray([current, limit, stop - start], dtype=np.float32),
        ])
        for kind, candidates in (("delete", selected_ranked), ("add", unselected_ranked)):
            valid_steps = min(MAX_STEPS, len(candidates), current - 1 if kind == "delete" else limit - current)
            for step in range(1, valid_steps + 1):
                local = int(candidates[step - 1])
                proposed = current - step if kind == "delete" else current + step
                needs = []
                exact_deltas = []
                entropies = []
                argmax_support = 0
                for probability in count_probabilities:
                    p = probability[oi]
                    if kind == "delete":
                        need = float(p[:proposed].sum())
                        argmax_support += int(np.argmax(p) + 1 <= proposed)
                    else:
                        need = float(p[proposed - 1:].sum())
                        argmax_support += int(np.argmax(p) + 1 >= proposed)
                    needs.append(need)
                    exact_deltas.append(float(p[proposed - 1] - p[current - 1]))
                    entropies.append(float(-(p * np.log(np.clip(p, 1e-9, 1))).sum()))
                local_sorted = np.sort(consensus)[::-1]
                node = score_matrix[local]
                rank = int(np.where(np.argsort(-consensus, kind="stable") == local)[0][0]) + 1
                boundary_gap = float(
                    consensus[local] - local_sorted[min(len(local_sorted) - 1, current)]
                )
                direction = 1.0 if kind == "add" else 0.0
                features = np.concatenate([
                    np.asarray([
                        direction, step, current, proposed, limit, rank, boundary_gap,
                        np.mean(needs), np.min(needs), np.max(needs), np.std(needs),
                        np.mean(exact_deltas), np.min(exact_deltas), argmax_support,
                        np.mean(entropies), np.std(entropies),
                    ], dtype=np.float32),
                    np.asarray(needs, dtype=np.float32),
                    np.asarray(exact_deltas, dtype=np.float32),
                    node.astype(np.float32), order_summary.astype(np.float32),
                    alarm_x[start + local].astype(np.float32),
                ])
                rows.append(features)
                meta.append({
                    "order_index": oi, "row": start + local, "local": local,
                    "kind": kind, "step": step, "current_count": current,
                    "proposed_count": proposed, "need_mean": float(np.mean(needs)),
                    "need_min": float(np.min(needs)), "argmax_support": argmax_support,
                    "node_scores": node.tolist(),
                })
                if labels is not None:
                    truth = int(labels[start + local])
                    targets.append(truth if kind == "add" else 1 - truth)
    return np.asarray(rows, dtype=np.float32), meta, np.asarray(targets, dtype=np.int8)


def crossfit(train_x, target, test_x, action_folds, sample_weights):
    oof_et = np.zeros(len(target), dtype=np.float32)
    oof_hgb = np.zeros(len(target), dtype=np.float32)
    test_et, test_hgb = [], []
    for fold in range(5):
        fit, valid = action_folds != fold, action_folds == fold
        et = ExtraTreesClassifier(
            n_estimators=800, min_samples_leaf=4, max_features=0.45,
            class_weight="balanced", n_jobs=-1, random_state=SEED + fold,
        )
        hgb = HistGradientBoostingClassifier(
            learning_rate=0.045, max_iter=260, max_leaf_nodes=15,
            min_samples_leaf=24, l2_regularization=1.5, random_state=SEED + fold,
        )
        et.fit(train_x[fit], target[fit], sample_weight=sample_weights[fit])
        hgb.fit(train_x[fit], target[fit], sample_weight=sample_weights[fit])
        oof_et[valid] = et.predict_proba(train_x[valid])[:, 1]
        oof_hgb[valid] = hgb.predict_proba(train_x[valid])[:, 1]
        test_et.append(et.predict_proba(test_x)[:, 1])
        test_hgb.append(hgb.predict_proba(test_x)[:, 1])
    oof = (oof_et + oof_hgb) / 2
    test = (np.mean(test_et, axis=0) + np.mean(test_hgb, axis=0)) / 2
    return oof.astype(np.float32), test.astype(np.float32), oof_et, oof_hgb


def precision_curves(scores, targets, meta, folds):
    report = {}
    for kind in ("all", "add", "delete"):
        mask = np.ones(len(scores), dtype=bool) if kind == "all" else np.asarray([m["kind"] == kind for m in meta])
        indices = np.flatnonzero(mask)
        indices = indices[np.argsort(-scores[indices], kind="stable")]
        rows = []
        for k in (5, 10, 15, 20, 30, 40, 60, 80, 100):
            chosen = indices[:min(k, len(indices))]
            if not len(chosen):
                continue
            fold_precision = []
            for fold in range(5):
                local = chosen[folds[chosen] == fold]
                if len(local):
                    fold_precision.append(float(targets[local].mean()))
            rows.append({
                "k": int(len(chosen)), "correct": int(targets[chosen].sum()),
                "precision": float(targets[chosen].mean()),
                "minimum_fold_precision": min(fold_precision) if fold_precision else None,
            })
        report[kind] = rows
    return report


def choose_unique_test_candidates(scores, meta, records, fixed_nodes, exact_orders):
    ranked = np.argsort(-scores, kind="stable")
    output, used_orders = [], set(exact_orders)
    for index in ranked:
        row = dict(meta[int(index)])
        order = records[row["order_index"]]
        key = (order["order_id"], order["alarms"][row["local"]]["rid"])
        if key in fixed_nodes or order["order_id"] in used_orders:
            continue
        used_orders.add(order["order_id"])
        row.update({
            "order_id": order["order_id"], "rid": key[1],
            "title": order["alarms"][row["local"]].get("title", ""),
            "ranker_probability": float(scores[index]),
        })
        output.append(row)
    return output


def emit_probe(name, candidates, exact_actions, order_ids, champion_roots, records_by_order):
    actions = [dict(action) for action in exact_actions]
    for i, row in enumerate(candidates, 1):
        actions.append({
            "action_id": f"{name}_{i:03d}", "order_id": row["order_id"],
            "remove_rids": [row["rid"]] if row["kind"] == "delete" else [],
            "add_rids": [row["rid"]] if row["kind"] == "add" else [],
            "source": "v38_boundary_action_ranker", "expected_gain": row["ranker_probability"],
            "evidence": row,
        })
    roots = apply_actions(champion_roots, records_by_order, actions)
    predictions = sum(map(len, roots.values()))
    path = OUT / "submissions" / f"{name}.csv"
    write_submission(path, order_ids, roots)
    fixed_gain = 1
    variable = len(candidates)
    possibilities = []
    if candidates and all(row["kind"] == "add" for row in candidates):
        for true_adds in range(variable + 1):
            tp = BASE_TP + fixed_gain + true_adds
            possibilities.append({"true_adds": true_adds, "tp": tp,
                                  "score": round(2 * tp / (TRUE_ROOTS + predictions), 9)})
    elif candidates and all(row["kind"] == "delete" for row in candidates):
        for deleted_true in range(variable + 1):
            tp = BASE_TP + fixed_gain - deleted_true
            possibilities.append({"deleted_true": deleted_true, "tp": tp,
                                  "score": round(2 * tp / (TRUE_ROOTS + predictions), 9)})
    manifest = {
        "probe_id": name, "baseline": str(CHAMPION), "path": str(path),
        "known_exact_gain": fixed_gain, "candidate_count": variable,
        "actions": actions, "predictions": predictions, "sha256": sha256(path),
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
    order_ids, champion_roots = load_submission(CHAMPION)
    test_base = selected_mask(records["test"], arrays["test_alarm_ptr"], champion_roots)
    train_node_scores = node_score_arrays(arrays, "train")
    test_node_scores = node_score_arrays(arrays, "test")
    train_base = exact_count_mask(train_node_scores[-1], arrays["train_alarm_ptr"], TRAIN_BUDGET)
    train_prob = [np.load(path) for path in probability_paths("oof")]
    test_prob = [np.load(path) for path in probability_paths("test")]
    train_x, train_meta, target = make_actions(
        arrays, "train", train_base, train_prob, train_node_scores,
        arrays["train_labels"].astype(np.int8),
    )
    test_x, test_meta, _ = make_actions(
        arrays, "test", test_base, test_prob, test_node_scores,
    )
    order_folds = arrays["train_station_folds"].astype(np.int8)
    action_folds = np.asarray([order_folds[row["order_index"]] for row in train_meta])
    order_weights = np.asarray([
        float(np.load(V33 / "importance_weights.npy")[int(start):int(stop)].mean())
        for start, stop in zip(arrays["train_alarm_ptr"][:-1], arrays["train_alarm_ptr"][1:])
    ])
    action_weights = np.asarray([order_weights[row["order_index"]] for row in train_meta])
    action_weights /= action_weights.mean()
    oof, test_score, oof_et, oof_hgb = crossfit(
        train_x, target, test_x, action_folds, action_weights
    )
    np.save(OUT / "oof_probability.npy", oof)
    np.save(OUT / "test_probability.npy", test_score)

    exact = read_json(EXACT_MANIFEST)
    exact_actions = exact["actions"]
    exact_orders = {action["order_id"] for action in exact_actions}
    fixed_report = read_json(V37 / "report.json")
    fixed_nodes = {(row["order_id"], row["rid"]) for row in fixed_report["fixed_labels"]}
    test_candidates = choose_unique_test_candidates(
        test_score, test_meta, records["test"], fixed_nodes, exact_orders
    )
    adds = [row for row in test_candidates if row["kind"] == "add"]
    deletes = [row for row in test_candidates if row["kind"] == "delete"]
    records_by_order = {row["order_id"]: row for row in records["test"]}
    probes = {}
    for kind, candidates in (("add", adds), ("delete", deletes)):
        for size in (8, 16, 24):
            subset = candidates[:size]
            if not subset:
                continue
            name = f"v38_{kind}_top{len(subset):02d}_plus_exact"
            probes[name] = emit_probe(
                name, subset, exact_actions, order_ids, champion_roots, records_by_order
            )
            if len(subset) < size:
                break
    curves = precision_curves(oof, target, train_meta, action_folds)
    report = {
        "version": "v38-boundary-action-ranker-1",
        "train_actions": len(train_meta), "test_actions": len(test_meta),
        "train_positive_rate": float(target.mean()),
        "oof_curves": curves,
        "oof_et_top20_precision": float(target[np.argsort(-oof_et)[:20]].mean()),
        "oof_hgb_top20_precision": float(target[np.argsort(-oof_hgb)[:20]].mean()),
        "test_unique_candidates": len(test_candidates),
        "test_add_candidates": len(adds), "test_delete_candidates": len(deletes),
        "top_test_candidates": test_candidates[:60], "probes": probes,
    }
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "train_actions": report["train_actions"], "test_actions": report["test_actions"],
        "oof_curves": curves, "test_add_candidates": len(adds),
        "test_delete_candidates": len(deletes),
        "top_test_candidates": test_candidates[:10],
        "probes": {name: {k: value[k] for k in ("path", "predictions", "sha256")}
                   for name, value in probes.items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
