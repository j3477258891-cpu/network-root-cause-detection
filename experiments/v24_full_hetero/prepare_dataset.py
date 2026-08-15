"""Build V24 complete heterogeneous per-order graphs without identity features."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np


VERSION = "v24-full-hetero-1"
NODE_CLASSES = (
    "Alarm",
    "RRU",
    "Cell",
    "BaseStation",
    "TransDevice",
    "TransBoard",
    "Board",
    "BBU",
    "Room",
    "RiPort",
    "NTransPort",
    "TransCircuit",
    "TransCircuitRoute",
    "UTransZPort",
    "UTransAPort",
)
NODE_CLASS_TO_ID = {name: index for index, name in enumerate(NODE_CLASSES)}
EDGE_CLASSES = ("dependOn", "generate", "causedBy")
EDGE_CLASS_TO_ID = {name: index for index, name in enumerate(EDGE_CLASSES)}
RELATION_COUNT = 7  # three forward, three reverse, one self-loop
MAX_PATH_NODES = 12
N_FOLDS = 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(value):
    if isinstance(value, list):
        return tuple(value)
    return value or ""


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_orders(root: Path, with_labels: bool) -> list[dict]:
    orders = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        order_id = directory.name
        topology = json.loads(
            (directory / f"{order_id}.log.topo.json").read_text(encoding="utf-8")
        )
        roots = set()
        if with_labels:
            root_data = json.loads(
                (directory / f"{order_id}.rootcause.json").read_text(encoding="utf-8")
            )
            roots = {node["@rid"] for node in root_data.get("rootcause", [])}
        orders.append({"id": order_id, "topology": topology, "roots": roots})
    return orders


def order_signature(order: dict):
    alarms = [
        node for node in order["topology"].get("nodes", []) if node.get("@class") == "Alarm"
    ]
    title_counts = Counter(scalar(node.get("title")) for node in alarms)
    target_titles = sorted(
        scalar(node.get("title")) for node in alarms if node.get("label") == "TargetAlarm"
    )
    return tuple(sorted(title_counts.items())), tuple(target_titles), len(alarms)


def grouped_folds(orders: list[dict]) -> tuple[np.ndarray, int, list[int]]:
    groups = defaultdict(list)
    for index, order in enumerate(orders):
        groups[order_signature(order)].append(index)
    sizes = [0] * N_FOLDS
    folds = np.zeros(len(orders), dtype=np.int8)
    ranked = sorted(
        groups.items(),
        key=lambda item: (
            -len(item[1]),
            hashlib.sha256(repr(item[0]).encode("utf-8")).hexdigest(),
        ),
    )
    for _, indices in ranked:
        fold = min(range(N_FOLDS), key=lambda value: (sizes[value], value))
        folds[indices] = fold
        sizes[fold] += len(indices)
    return folds, len(groups), sizes


def read_submission(path: Path) -> dict[str, set[str]]:
    rows = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["order_id", "output"]:
            raise ValueError(f"invalid submission columns: {reader.fieldnames}")
        for row in reader:
            rows[row["order_id"]] = {
                node["@rid"] for node in json.loads(row["output"])["rootcause"]
            }
    return rows


def shortest_distances(adjacency: list[list[int]], starts: list[int]) -> np.ndarray:
    unreachable = 13
    distances = np.full(len(adjacency), unreachable, dtype=np.float32)
    queue = deque()
    for start in starts:
        distances[start] = 0
        queue.append(start)
    while queue:
        current = queue.popleft()
        if distances[current] >= 12:
            continue
        for neighbor in adjacency[current]:
            if distances[neighbor] > distances[current] + 1:
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)
    return distances


def invert_relation(relation: int) -> int:
    return relation + 3 if relation < 3 else relation - 3


def topology_arrays(order: dict, alarm_feature_offset: int) -> dict:
    topology = order["topology"]
    nodes = topology.get("nodes", [])
    rid_to_index = {node.get("@rid"): index for index, node in enumerate(nodes)}
    outgoing = [[] for _ in nodes]
    incoming = [[] for _ in nodes]
    undirected = [[] for _ in nodes]
    typed_adjacency = [[] for _ in nodes]
    edge_src, edge_dst, edge_type = [], [], []
    missing_endpoints = 0

    for edge in topology.get("edges", []):
        source = rid_to_index.get(edge.get("in"))
        target = rid_to_index.get(edge.get("out"))
        relation = EDGE_CLASS_TO_ID.get(edge.get("@class"))
        if source is None or target is None or relation is None:
            missing_endpoints += 1
            continue
        outgoing[source].append(target)
        incoming[target].append(source)
        undirected[source].append(target)
        undirected[target].append(source)
        typed_adjacency[source].append((target, relation))
        typed_adjacency[target].append((source, relation + 3))
        edge_src.extend((source, target))
        edge_dst.extend((target, source))
        edge_type.extend((relation, relation + 3))
    for index in range(len(nodes)):
        edge_src.append(index)
        edge_dst.append(index)
        edge_type.append(6)

    alarm_indices = [index for index, node in enumerate(nodes) if node.get("@class") == "Alarm"]
    target_indices = [index for index in alarm_indices if nodes[index].get("label") == "TargetAlarm"]
    if not alarm_indices:
        raise ValueError(f"order has no Alarm nodes: {order['id']}")
    if not target_indices:
        target_indices = alarm_indices[:1]

    from_target = shortest_distances(outgoing, target_indices)
    to_target = shortest_distances(incoming, target_indices)
    undirected_target = shortest_distances(undirected, target_indices)
    fault_time = safe_float(topology.get("time"), 0.0)
    node_times = np.asarray([safe_float(node.get("time"), fault_time) for node in nodes])
    time_min = float(node_times.min()) if len(node_times) else fault_time
    time_max = float(node_times.max()) if len(node_times) else fault_time
    time_span = max(time_max - time_min, 1.0)

    numeric = np.zeros((len(nodes), 18), dtype=np.float32)
    type_ids = np.zeros(len(nodes), dtype=np.int16)
    for index, node in enumerate(nodes):
        type_ids[index] = NODE_CLASS_TO_ID.get(node.get("@class"), 0)
        in_by_type = [0, 0, 0]
        out_by_type = [0, 0, 0]
        for neighbor, relation in typed_adjacency[index]:
            del neighbor
            if relation < 3:
                out_by_type[relation] += 1
            else:
                in_by_type[relation - 3] += 1
        timeline = node.get("timeLists", [])
        if not isinstance(timeline, list):
            timeline = []
        timeline_values = np.asarray(
            [safe_float(value) for value in timeline[:6]], dtype=np.float32
        )
        numeric[index] = np.asarray(
            [
                math.log1p(len(incoming[index])),
                math.log1p(len(outgoing[index])),
                *[math.log1p(value) for value in in_by_type],
                *[math.log1p(value) for value in out_by_type],
                min(float(from_target[index]), 13.0) / 13.0,
                min(float(to_target[index]), 13.0) / 13.0,
                min(float(undirected_target[index]), 13.0) / 13.0,
                float(node.get("@class") == "Alarm"),
                float(node.get("label") == "TargetAlarm"),
                (node_times[index] - time_min) / time_span,
                math.log1p(abs(node_times[index] - fault_time)),
                float(timeline_values.mean()) if len(timeline_values) else 0.0,
                math.log1p(len(nodes)),
                math.log1p(len(topology.get("edges", []))),
            ],
            dtype=np.float32,
        )

    target_set = set(target_indices)
    parent = np.full(len(nodes), -1, dtype=np.int32)
    relation_to_parent = np.full(len(nodes), -1, dtype=np.int8)
    queue = deque(target_indices)
    seen = set(target_indices)
    while queue:
        current = queue.popleft()
        for neighbor, current_to_neighbor in typed_adjacency[current]:
            if neighbor in seen:
                continue
            seen.add(neighbor)
            parent[neighbor] = current
            relation_to_parent[neighbor] = invert_relation(current_to_neighbor)
            queue.append(neighbor)

    path_node_types = np.full((len(alarm_indices), MAX_PATH_NODES), -1, dtype=np.int16)
    path_edge_types = np.full((len(alarm_indices), MAX_PATH_NODES - 1), -1, dtype=np.int8)
    path_lengths = np.ones(len(alarm_indices), dtype=np.int8)
    for local_alarm, node_index in enumerate(alarm_indices):
        path_nodes = [node_index]
        path_edges = []
        current = node_index
        while current not in target_set and parent[current] >= 0 and len(path_nodes) < MAX_PATH_NODES:
            path_edges.append(int(relation_to_parent[current]))
            current = int(parent[current])
            path_nodes.append(current)
        path_lengths[local_alarm] = len(path_nodes)
        path_node_types[local_alarm, : len(path_nodes)] = type_ids[path_nodes]
        if path_edges:
            path_edge_types[local_alarm, : len(path_edges)] = path_edges

    return {
        "node_type": type_ids,
        "node_numeric": numeric,
        "edge_src": np.asarray(edge_src, dtype=np.int32),
        "edge_dst": np.asarray(edge_dst, dtype=np.int32),
        "edge_type": np.asarray(edge_type, dtype=np.int8),
        "alarm_node_index": np.asarray(alarm_indices, dtype=np.int32),
        "alarm_feature_index": np.arange(
            alarm_feature_offset, alarm_feature_offset + len(alarm_indices), dtype=np.int32
        ),
        "path_node_type": path_node_types,
        "path_edge_type": path_edge_types,
        "path_length": path_lengths,
        "missing_endpoints": missing_endpoints,
        "alarms": [nodes[index] for index in alarm_indices],
    }


def build_split(orders: list[dict]) -> tuple[dict[str, np.ndarray], list[dict], int]:
    node_ptr = [0]
    edge_ptr = [0]
    alarm_ptr = [0]
    values = defaultdict(list)
    metadata = []
    missing_endpoints = 0
    alarm_offset = 0
    for order_index, order in enumerate(orders):
        graph = topology_arrays(order, alarm_offset)
        node_offset = node_ptr[-1]
        values["node_type"].append(graph["node_type"])
        values["node_numeric"].append(graph["node_numeric"])
        values["edge_src"].append(graph["edge_src"] + node_offset)
        values["edge_dst"].append(graph["edge_dst"] + node_offset)
        values["edge_type"].append(graph["edge_type"])
        values["alarm_node_index"].append(graph["alarm_node_index"] + node_offset)
        values["alarm_feature_index"].append(graph["alarm_feature_index"])
        values["path_node_type"].append(graph["path_node_type"])
        values["path_edge_type"].append(graph["path_edge_type"])
        values["path_length"].append(graph["path_length"])
        node_ptr.append(node_offset + len(graph["node_type"]))
        edge_ptr.append(edge_ptr[-1] + len(graph["edge_src"]))
        alarm_ptr.append(alarm_ptr[-1] + len(graph["alarm_node_index"]))
        alarm_offset = alarm_ptr[-1]
        missing_endpoints += graph["missing_endpoints"]
        metadata.append(
            {
                "order_id": order["id"],
                "alarms": [
                    {
                        "@rid": node["@rid"],
                        "title": node.get("title", ""),
                        "location": node.get("location", ""),
                        "reason": node.get("reason", ""),
                    }
                    for node in graph["alarms"]
                ],
            }
        )
        if (order_index + 1) % 200 == 0:
            print(f"build: {order_index + 1}/{len(orders)}", flush=True)
    arrays = {name: np.concatenate(parts) for name, parts in values.items()}
    arrays.update(
        node_ptr=np.asarray(node_ptr, dtype=np.int64),
        edge_ptr=np.asarray(edge_ptr, dtype=np.int64),
        alarm_ptr=np.asarray(alarm_ptr, dtype=np.int64),
    )
    return arrays, metadata, missing_endpoints


def normalize(train: np.ndarray, test: np.ndarray, min_std=0.05):
    mean = train.mean(axis=0, dtype=np.float64)
    std = train.std(axis=0, dtype=np.float64)
    std = np.maximum(std, min_std)
    return (
        np.clip((train - mean) / std, -10.0, 10.0).astype(np.float32),
        np.clip((test - mean) / std, -10.0, 10.0).astype(np.float32),
        mean.astype(np.float32),
        std.astype(np.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    args.output.mkdir(parents=True, exist_ok=True)

    train_orders = load_orders(root / "train", True)
    test_orders = load_orders(root / "test", False)
    train_graph, train_meta, train_missing = build_split(train_orders)
    test_graph, test_meta, test_missing = build_split(test_orders)

    train_alarm_x = np.load(root / "experiments/v16/v16_train_features.npy")
    test_alarm_x = np.load(root / "experiments/v16/v16_test_features.npy")
    if len(train_alarm_x) != train_graph["alarm_ptr"][-1] or len(test_alarm_x) != test_graph["alarm_ptr"][-1]:
        raise ValueError("V16 Alarm feature rows do not match V24 Alarm ordering")
    train_alarm_x, test_alarm_x, alarm_mean, alarm_std = normalize(
        train_alarm_x, test_alarm_x
    )
    train_node_x, test_node_x, node_mean, node_std = normalize(
        train_graph.pop("node_numeric"), test_graph.pop("node_numeric")
    )
    train_graph["node_numeric"] = train_node_x
    test_graph["node_numeric"] = test_node_x

    train_v11 = (
        0.25 * np.load(root / "codexgz/v11/v11_oof_context.npy")
        + 0.75 * np.load(root / "codexgz/v11/v11_oof_meta.npy")
    ).astype(np.float32)
    test_v11 = np.load(root / "codexgz/v11/v11_test_scores.npy").astype(np.float32)
    train_labels = np.asarray(
        [
            int(alarm["@rid"] in order["roots"])
            for order, item in zip(train_orders, train_meta)
            for alarm in item["alarms"]
        ],
        dtype=np.int8,
    )
    train_counts = np.asarray([len(order["roots"]) for order in train_orders], dtype=np.int8)
    folds, group_count, fold_sizes = grouped_folds(train_orders)
    champion_path = root / "experiments/submissions/champion_0.906324_day01_probe01_v11_full.csv"
    champion = read_submission(champion_path)
    test_champion_mask = np.asarray(
        [
            alarm["@rid"] in champion[item["order_id"]]
            for item in test_meta
            for alarm in item["alarms"]
        ],
        dtype=np.int8,
    )

    arrays = {
        **{f"train_{key}": value for key, value in train_graph.items()},
        **{f"test_{key}": value for key, value in test_graph.items()},
        "train_alarm_x": train_alarm_x,
        "test_alarm_x": test_alarm_x,
        "alarm_feature_mean": alarm_mean,
        "alarm_feature_std": alarm_std,
        "node_feature_mean": node_mean,
        "node_feature_std": node_std,
        "train_v11": train_v11,
        "test_v11": test_v11,
        "train_labels": train_labels,
        "train_counts": train_counts,
        "train_folds": folds,
        "test_champion_mask": test_champion_mask,
    }
    npz_path = args.output / "rootcause_v24_full_hetero.npz"
    np.savez_compressed(npz_path, **arrays)

    metadata = {
        "version": VERSION,
        "node_classes": NODE_CLASSES,
        "edge_classes": EDGE_CLASSES,
        "relation_count": RELATION_COUNT,
        "max_path_nodes": MAX_PATH_NODES,
        "group_count": group_count,
        "fold_sizes": fold_sizes,
        "train": train_meta,
        "test": test_meta,
        "champion": str(champion_path),
        "champion_sha256": sha256(champion_path),
        "excluded_identity_features": ["@rid", "zh_label", "title", "location", "IP", "device_id"],
    }
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "version": VERSION,
        "design": "disjoint complete per-order heterogeneous graphs",
        "files": {
            npz_path.name: {"bytes": npz_path.stat().st_size, "sha256": sha256(npz_path)},
            metadata_path.name: {
                "bytes": metadata_path.stat().st_size,
                "sha256": sha256(metadata_path),
            },
        },
        "counts": {
            "train_orders": len(train_orders),
            "test_orders": len(test_orders),
            "train_nodes": int(train_graph["node_ptr"][-1]),
            "test_nodes": int(test_graph["node_ptr"][-1]),
            "train_edges_with_reverse_self": int(train_graph["edge_ptr"][-1]),
            "test_edges_with_reverse_self": int(test_graph["edge_ptr"][-1]),
            "train_alarms": len(train_labels),
            "test_alarms": len(test_v11),
            "train_positive": int(train_labels.sum()),
            "filtered_missing_endpoints": train_missing + test_missing,
        },
        "dimensions": {
            "alarm_features": int(train_alarm_x.shape[1]),
            "node_numeric": int(train_node_x.shape[1]),
            "node_types": len(NODE_CLASSES),
            "relations": RELATION_COUNT,
            "max_path_nodes": MAX_PATH_NODES,
        },
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), **manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
