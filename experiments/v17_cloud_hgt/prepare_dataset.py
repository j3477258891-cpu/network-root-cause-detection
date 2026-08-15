"""Build the compact V17 alarm graph dataset from the raw topology files."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np


ROOT = Path(r"D:\zgyidong")
V16_DIR = ROOT / "experiments" / "v16"
V11_DIR = ROOT / "codexgz" / "v11"
CHAMPION = ROOT / "experiments" / "submissions" / "champion_0.906324_day01_probe01_v11_full.csv"
LOCKS = V11_DIR / "v11_constrained_report.json"
EXCLUSIONS = ROOT / "experiments" / "v15" / "v15b_exclusions.json"

sys.path.insert(0, str(V16_DIR))
import v16_structure_signal as v16


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def order_ptr(orders):
    ptr = [0]
    for order in orders:
        ptr.append(ptr[-1] + len(order["alarms"]))
    return np.asarray(ptr, dtype=np.int64)


def capped_alarm_edges(order, max_depth=3):
    nodes = order["topology"].get("nodes", [])
    rid_to_index = {node.get("@rid"): index for index, node in enumerate(nodes)}
    outgoing = [set() for _ in nodes]
    undirected = [set() for _ in nodes]
    for edge in order["topology"].get("edges", []):
        source = rid_to_index.get(edge.get("in"))
        target = rid_to_index.get(edge.get("out"))
        if source is None or target is None:
            continue
        outgoing[source].add(target)
        undirected[source].add(target)
        undirected[target].add(source)
    alarm_global = [rid_to_index[node["@rid"]] for node in order["alarms"]]
    alarm_local = {value: index for index, value in enumerate(alarm_global)}
    edges = []
    for source_local, source_global in enumerate(alarm_global):
        edges.append((source_local, source_local, 0))
        distances = {source_global: 0}
        queue = deque([source_global])
        while queue:
            current = queue.popleft()
            depth = distances[current]
            if depth >= max_depth:
                continue
            for neighbor in undirected[current]:
                if neighbor not in distances:
                    distances[neighbor] = depth + 1
                    queue.append(neighbor)
        for target_global, depth in distances.items():
            if target_global == source_global or target_global not in alarm_local:
                continue
            target_local = alarm_local[target_global]
            if depth == 1 and target_global in outgoing[source_global]:
                relation = 1
            elif depth == 1 and source_global in outgoing[target_global]:
                relation = 2
            elif depth == 2:
                relation = 3
            else:
                relation = 4
            edges.append((source_local, target_local, relation))
    if len(alarm_global) > 1 and len(edges) == len(alarm_global):
        for source in range(len(alarm_global)):
            for target in range(len(alarm_global)):
                if source != target:
                    edges.append((source, target, 4))
    return edges


def graph_arrays(orders):
    ptr = order_ptr(orders)
    sources, targets, relations = [], [], []
    edge_ptr = [0]
    for index, order in enumerate(orders):
        offset = int(ptr[index])
        edges = capped_alarm_edges(order)
        sources.extend(offset + source for source, _, _ in edges)
        targets.extend(offset + target for _, target, _ in edges)
        relations.extend(relation for _, _, relation in edges)
        edge_ptr.append(len(sources))
        if (index + 1) % 200 == 0:
            print(f"edges {index + 1}/{len(orders)}", flush=True)
    return (
        ptr,
        np.asarray(edge_ptr, dtype=np.int64),
        np.asarray(sources, dtype=np.int64),
        np.asarray(targets, dtype=np.int64),
        np.asarray(relations, dtype=np.int8),
    )


def normalize_features(train_x, test_x):
    mean = np.mean(train_x, axis=0, dtype=np.float64)
    std = np.std(train_x, axis=0, dtype=np.float64)
    std = np.maximum(std, 0.05)
    train = np.clip((train_x - mean) / std, -10.0, 10.0).astype(np.float32)
    test = np.clip((test_x - mean) / std, -10.0, 10.0).astype(np.float32)
    return train, test, mean.astype(np.float32), std.astype(np.float32)


def submission_metadata(test_orders):
    return [
        {
            "order_id": order["id"],
            "nodes": [
                {
                    "@rid": node["@rid"],
                    "title": node.get("title", ""),
                    "location": node.get("location", ""),
                    "reason": node.get("reason", ""),
                }
                for node in order["alarms"]
            ],
        }
        for order in test_orders
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "cloud_dataset")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    train_orders = v16.load_orders(v16.TRAIN_DIR, True)
    test_orders = v16.load_orders(v16.TEST_DIR, False)
    train_x = np.load(V16_DIR / "v16_train_features.npy")
    test_x = np.load(V16_DIR / "v16_test_features.npy")
    train_x, test_x, feature_mean, feature_std = normalize_features(train_x, test_x)
    train_graph = graph_arrays(train_orders)
    test_graph = graph_arrays(test_orders)
    train_labels = np.asarray(
        [int(node["@rid"] in order["roots"]) for order in train_orders for node in order["alarms"]],
        dtype=np.int8,
    )
    train_counts = np.asarray([len(order["roots"]) for order in train_orders], dtype=np.int8)
    folds, group_count, fold_sizes = v16.grouped_folds(train_orders)
    train_v11 = 0.25 * np.load(V11_DIR / "v11_oof_context.npy") + 0.75 * np.load(
        V11_DIR / "v11_oof_meta.npy"
    )
    test_v11 = np.load(V11_DIR / "v11_test_scores.npy")
    champion_rows = v16.read_submission(CHAMPION)
    test_champion_mask = np.asarray(
        [
            node["@rid"] in {item["@rid"] for item in champion_rows[order["id"]]}
            for order in test_orders
            for node in order["alarms"]
        ],
        dtype=np.int8,
    )

    npz_path = args.output / "rootcause_v17_graphs.npz"
    np.savez_compressed(
        npz_path,
        train_x=train_x,
        test_x=test_x,
        feature_mean=feature_mean,
        feature_std=feature_std,
        train_labels=train_labels,
        train_counts=train_counts,
        train_folds=folds,
        train_v11=train_v11.astype(np.float32),
        test_v11=test_v11.astype(np.float32),
        test_champion_mask=test_champion_mask,
        train_order_ptr=train_graph[0],
        train_edge_ptr=train_graph[1],
        train_edge_src=train_graph[2],
        train_edge_dst=train_graph[3],
        train_edge_type=train_graph[4],
        test_order_ptr=test_graph[0],
        test_edge_ptr=test_graph[1],
        test_edge_src=test_graph[2],
        test_edge_dst=test_graph[3],
        test_edge_type=test_graph[4],
    )
    meta_path = args.output / "metadata.json"
    metadata = {
        "version": "rootcause-v17-v1",
        "train_orders": [order["id"] for order in train_orders],
        "train_template_sha": [
            hashlib.sha256(repr(v16.order_signature(order)).encode("utf-8")).hexdigest()
            for order in train_orders
        ],
        "test": submission_metadata(test_orders),
        "test_template_sha": [
            hashlib.sha256(repr(v16.order_signature(order)).encode("utf-8")).hexdigest()
            for order in test_orders
        ],
        "group_count": group_count,
        "fold_sizes": fold_sizes,
        "feature_dim": int(train_x.shape[1]),
        "relation_count": 5,
        "champion": str(CHAMPION),
        "champion_sha256": sha256(CHAMPION),
        "locks": json.loads(LOCKS.read_text(encoding="utf-8")),
        "exclusions": json.loads(EXCLUSIONS.read_text(encoding="utf-8")),
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "version": "rootcause-v17-v1",
        "files": {
            npz_path.name: {"bytes": npz_path.stat().st_size, "sha256": sha256(npz_path)},
            meta_path.name: {"bytes": meta_path.stat().st_size, "sha256": sha256(meta_path)},
        },
        "counts": {
            "train_orders": len(train_orders),
            "test_orders": len(test_orders),
            "train_nodes": int(train_x.shape[0]),
            "test_nodes": int(test_x.shape[0]),
            "train_positive": int(np.sum(train_labels)),
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
