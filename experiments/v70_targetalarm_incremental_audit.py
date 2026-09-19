"""Strict incremental audit for raw ``TargetAlarm`` topology evidence.

V16 already contains graph features, but V30 only consumes V16 as one member of
a broad score ensemble.  This audit asks the narrower question that matters for
deployment: does an explicit TargetAlarm-aware reranker improve the V30 ranking
when both the reranker and V30 score for an order exclude that order's station?

The true root count is used *only* to separate ranking quality from count
allocation.  No CSV is produced by this script.
"""

from __future__ import annotations

import gzip
import json
import sys
from collections import deque
from pathlib import Path

ROOT = Path(r"D:\zgyidong")
if str(ROOT / ".deps") not in sys.path:
    sys.path.insert(0, str(ROOT / ".deps"))

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier


DATA = ROOT / "experiments/v25_semantic_router/cloud_dataset/v25_semantic_router.npz"
RECORDS = ROOT / "experiments/v25_semantic_router/cloud_dataset/semantic_records.json.gz"
V30 = ROOT / "experiments/v30_meta_stack"
OUT = ROOT / "experiments/v70_targetalarm_incremental"
MAX_ROOTS = 8
SEED = 20260820


def norm(value: object) -> str:
    return " ".join(str(value or "").lower().split())


def relative_score_features(scores: np.ndarray, ptr: np.ndarray) -> np.ndarray:
    """Per-order score geometry, so raw probabilities are never the sole cue."""
    output = np.zeros((len(scores), 4), dtype=np.float32)
    for start, stop in zip(ptr[:-1], ptr[1:]):
        start, stop = int(start), int(stop)
        local = scores[start:stop]
        order = np.argsort(-local, kind="stable")
        ranks = np.empty(len(local), dtype=np.float32)
        ranks[order] = np.arange(len(local), dtype=np.float32)
        scale = max(len(local) - 1, 1)
        output[start:stop, 0] = ranks / scale
        output[start:stop, 1] = (local - local.mean()) / (local.std() + 1e-6)
        output[start:stop, 2] = local.max() - local
        output[start:stop, 3] = local - local.min()
    return output


def shortest(adjacency: list[set[int]], starts: set[int]) -> np.ndarray:
    unreachable = len(adjacency) + 1
    result = np.full(len(adjacency), unreachable, dtype=np.int16)
    queue: deque[int] = deque()
    for start in starts:
        result[start] = 0
        queue.append(start)
    while queue:
        current = queue.popleft()
        for child in adjacency[current]:
            if result[child] > result[current] + 1:
                result[child] = result[current] + 1
                queue.append(child)
    return result


def topology_features(order_id: str, expected_rids: list[str], split: str) -> np.ndarray:
    path = ROOT / split / order_id / f"{order_id}.log.topo.json"
    topology = json.loads(path.read_text(encoding="utf-8"))
    nodes = topology.get("nodes", [])
    rid_index = {str(node.get("@rid")): index for index, node in enumerate(nodes)}
    outgoing = [set() for _ in nodes]
    incoming = [set() for _ in nodes]
    undirected = [set() for _ in nodes]
    for edge in topology.get("edges", []):
        source = rid_index.get(str(edge.get("in")))
        target = rid_index.get(str(edge.get("out")))
        if source is None or target is None:
            continue
        outgoing[source].add(target)
        incoming[target].add(source)
        undirected[source].add(target)
        undirected[target].add(source)
    alarm_indices = [index for index, node in enumerate(nodes) if node.get("@class") == "Alarm"]
    target_set = {index for index in alarm_indices if nodes[index].get("label") == "TargetAlarm"}
    from_target = shortest(outgoing, target_set)
    to_target = shortest(incoming, target_set)
    undirected_target = shortest(undirected, target_set)
    title_count: dict[str, int] = {}
    reason_count: dict[str, int] = {}
    device_count: dict[str, int] = {}
    vendor_count: dict[str, int] = {}
    location_count: dict[str, int] = {}
    for index in target_set:
        node = nodes[index]
        for table, key in ((title_count, "title"), (reason_count, "reason"),
                           (device_count, "device"), (vendor_count, "vendor"),
                           (location_count, "location")):
            value = norm(node.get(key))
            if value:
                table[value] = table.get(value, 0) + 1
    unreachable = len(nodes) + 1
    rows: dict[str, np.ndarray] = {}
    for index in alarm_indices:
        node = nodes[index]
        out, inc, und = outgoing[index], incoming[index], undirected[index]
        target_neighbors_out = sum(item in target_set for item in out)
        target_neighbors_in = sum(item in target_set for item in inc)
        target_neighbors_und = sum(item in target_set for item in und)
        rows[str(node.get("@rid"))] = np.asarray([
            float(index in target_set),
            float(len(target_set)),
            float(len(alarm_indices)),
            float(title_count.get(norm(node.get("title")), 0)),
            float(reason_count.get(norm(node.get("reason")), 0)),
            float(device_count.get(norm(node.get("device")), 0)),
            float(vendor_count.get(norm(node.get("vendor")), 0)),
            float(location_count.get(norm(node.get("location")), 0)),
            float(target_neighbors_out),
            float(target_neighbors_in),
            float(target_neighbors_und),
            float(np.log1p(min(int(from_target[index]), unreachable))),
            float(np.log1p(min(int(to_target[index]), unreachable))),
            float(np.log1p(min(int(undirected_target[index]), unreachable))),
            float(from_target[index] <= len(nodes)),
            float(to_target[index] <= len(nodes)),
        ], dtype=np.float32)
    if set(rows) != set(expected_rids):
        raise ValueError(f"alarm alignment mismatch: {order_id}")
    return np.vstack([rows[rid] for rid in expected_rids])


def build_topology_matrix(records: list[dict], split: str) -> np.ndarray:
    parts = []
    for index, record in enumerate(records):
        parts.append(topology_features(
            record["order_id"], [str(alarm["rid"]) for alarm in record["alarms"]], split
        ))
        if (index + 1) % 250 == 0:
            print(f"{split}: extracted {index + 1}/{len(records)} orders", flush=True)
    return np.vstack(parts).astype(np.float32)


def fixed_order_count_mask(scores: np.ndarray, ptr: np.ndarray, labels: np.ndarray) -> np.ndarray:
    result = np.zeros(len(scores), dtype=bool)
    for start, stop in zip(ptr[:-1], ptr[1:]):
        start, stop = int(start), int(stop)
        count = min(MAX_ROOTS, int(labels[start:stop].sum()))
        local = np.argsort(-scores[start:stop], kind="stable")[:count]
        result[start + local] = True
    return result


def metrics(scores: np.ndarray, base: np.ndarray, labels: np.ndarray,
            ptr: np.ndarray, folds: np.ndarray) -> dict:
    proposed = fixed_order_count_mask(scores, ptr, labels)
    order_rows = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
    deltas = []
    for fold in range(5):
        mask = folds[order_rows] == fold
        deltas.append(int((proposed[mask] & labels[mask]).sum() - (base[mask] & labels[mask]).sum()))
    changed_orders = 0
    for start, stop in zip(ptr[:-1], ptr[1:]):
        if np.any(proposed[int(start):int(stop)] != base[int(start):int(stop)]):
            changed_orders += 1
    return {
        "delta_tp": int((proposed & labels).sum() - (base & labels).sum()),
        "station_fold_delta_tp": deltas,
        "changed_orders": changed_orders,
    }


def crossfit(kind: str, train_x: np.ndarray, labels: np.ndarray, test_x: np.ndarray,
             order_rows: np.ndarray, folds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    oof = np.zeros(len(labels), dtype=np.float32)
    test_predictions = []
    for fold in range(5):
        train_mask = folds[order_rows] != fold
        valid_mask = ~train_mask
        if kind == "target_only":
            model = ExtraTreesClassifier(
                n_estimators=500, min_samples_leaf=8, max_features=0.8,
                class_weight="balanced", n_jobs=-1, random_state=SEED + fold,
            )
        else:
            model = HistGradientBoostingClassifier(
                max_iter=220, learning_rate=0.035, max_leaf_nodes=15,
                min_samples_leaf=30, l2_regularization=3.0,
                class_weight="balanced", random_state=SEED + fold,
            )
        model.fit(train_x[train_mask], labels[train_mask])
        oof[valid_mask] = model.predict_proba(train_x[valid_mask])[:, 1]
        test_predictions.append(model.predict_proba(test_x)[:, 1])
    return oof, np.mean(test_predictions, axis=0).astype(np.float32)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(DATA, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    with gzip.open(RECORDS, "rt", encoding="utf-8") as handle:
        records = json.load(handle)
    train_records, test_records = records["train"], records["test"]
    topo_train_path, topo_test_path = OUT / "targetalarm_train.npy", OUT / "targetalarm_test.npy"
    topo_train = np.load(topo_train_path) if topo_train_path.exists() else build_topology_matrix(train_records, "train")
    topo_test = np.load(topo_test_path) if topo_test_path.exists() else build_topology_matrix(test_records, "test")
    if not topo_train_path.exists(): np.save(topo_train_path, topo_train)
    if not topo_test_path.exists(): np.save(topo_test_path, topo_test)
    ptr = arrays["train_alarm_ptr"]
    labels = arrays["train_labels"].astype(bool)
    folds = arrays["train_station_folds"].astype(np.int8)
    order_rows = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
    v30_train = np.load(V30 / "station_extra_trees_oof.npy").astype(np.float32)
    v30_test = np.load(V30 / "station_extra_trees_test.npy").astype(np.float32)
    if len(topo_train) != len(labels) or len(topo_test) != len(v30_test):
        raise ValueError("feature matrix alignment differs from V25")
    base = fixed_order_count_mask(v30_train, ptr, labels)
    feature_sets = {
        "target_only": (topo_train, topo_test),
        "v30_plus_target": (
            np.column_stack([topo_train, v30_train, relative_score_features(v30_train, ptr)]),
            np.column_stack([topo_test, v30_test, relative_score_features(v30_test, arrays["test_alarm_ptr"])]),
        ),
    }
    results: dict[str, object] = {
        "base_v30_tp": int((base & labels).sum()),
        "target_feature_count": int(topo_train.shape[1]),
        "methods": {},
    }
    for name, (x_train, x_test) in feature_sets.items():
        oof, test = crossfit(name, x_train, labels, x_test, order_rows, folds)
        np.save(OUT / f"{name}_oof.npy", oof)
        np.save(OUT / f"{name}_test.npy", test)
        scan = {}
        for weight in np.linspace(0.0, 1.0, 11):
            blended = (1.0 - weight) * v30_train + weight * oof
            scan[f"{weight:.1f}"] = metrics(blended, base, labels, ptr, folds)
        results["methods"][name] = {"blend_scan": scan}
    eligible = []
    for name, value in results["methods"].items():
        for weight, row in value["blend_scan"].items():
            if row["delta_tp"] > 0 and min(row["station_fold_delta_tp"]) >= 0:
                eligible.append({"method": name, "weight": weight, **row})
    eligible.sort(key=lambda item: (item["delta_tp"], min(item["station_fold_delta_tp"])), reverse=True)
    results.update({
        "version": "v70-targetalarm-incremental-audit-1",
        "method": "Station-cross-fitted TargetAlarm rerankers versus station-cross-fitted V30; oracle per-order count fixes count allocation.",
        "warning": "No score or test array is a submission candidate. A count policy, action catalog, and online probe are required before deployment.",
        "eligible_nonnegative_station_candidates": eligible,
        "gate": {
            "requires_positive_total_delta": True,
            "requires_nonnegative_every_station_fold": True,
            "passed": bool(eligible),
        },
    })
    (OUT / "report.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"base_v30_tp": results["base_v30_tp"], "eligible": eligible, "gate": results["gate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
