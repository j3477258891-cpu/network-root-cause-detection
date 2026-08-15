"""Prepare the compact V25 semantic dataset from raw orders and V24 arrays."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from v25_common import (
    balanced_group_folds,
    clean_text,
    connected_group_folds,
    normalized_location,
    order_signature,
    parse_addinfo,
    read_json,
    station_ids,
    time_summary,
    write_json,
)


ROOT = Path(r"D:\zgyidong")


def load_split(directory: Path, with_labels: bool) -> list[dict]:
    output = []
    for order_dir in sorted(path for path in directory.iterdir() if path.is_dir()):
        order_id = order_dir.name
        topology = read_json(order_dir / f"{order_id}.log.topo.json")
        nodes = topology.get("nodes", [])
        alarms = [node for node in nodes if node.get("@class") == "Alarm"]
        roots = set()
        if with_labels:
            root_data = read_json(order_dir / f"{order_id}.rootcause.json")
            roots = {node["@rid"] for node in root_data.get("rootcause", [])}
        rid_to_node = {node.get("@rid"): node for node in nodes}
        adjacent = {node.get("@rid"): [] for node in alarms}
        for edge in topology.get("edges", []):
            left, right = rid_to_node.get(edge.get("in")), rid_to_node.get(edge.get("out"))
            if left and right and left.get("@class") == right.get("@class") == "Alarm":
                adjacent[left["@rid"]].append(clean_text(right.get("title"), 120))
                adjacent[right["@rid"]].append(clean_text(left.get("title"), 120))
        target_text = " | ".join(
            f"{clean_text(node.get('title'), 160)}:{clean_text(node.get('reason'), 260)}"
            for node in alarms
            if node.get("label") == "TargetAlarm"
        )[:1200]
        alarm_records = []
        all_stations = set()
        for node in alarms:
            info = parse_addinfo(node.get("addInfo"))
            ids = station_ids(node.get("location")) + station_ids(node.get("addInfo"))
            all_stations.update(ids)
            alarm_records.append(
                {
                    "rid": node["@rid"],
                    "title": clean_text(node.get("title"), 240),
                    "reason": clean_text(node.get("reason"), 600),
                    "location": normalized_location(node.get("location")),
                    "raw_location": str(node.get("location", "")),
                    "device": clean_text(node.get("device"), 160),
                    "vendor": clean_text(node.get("vendor"), 80),
                    "device_type": info["device_type"],
                    "board_type": info["board_type"],
                    "cause": info["cause"],
                    "radio": info["radio"],
                    "deployment": info["deployment"],
                    "label": str(node.get("label", "")),
                    "timeline": time_summary(node),
                    "target_summary": target_text,
                    "neighbor_titles": sorted(set(adjacent[node["@rid"]]))[:8],
                    "is_root": int(node["@rid"] in roots) if with_labels else None,
                    "source": {
                        "title": node.get("title", ""),
                        "location": node.get("location", ""),
                        "reason": node.get("reason", ""),
                    },
                }
            )
        output.append(
            {
                "order_id": order_id,
                "fault_time": int(topology.get("time", 0) or 0),
                "station_ids": sorted(all_stations),
                "signature": order_signature(alarms),
                "alarms": alarm_records,
            }
        )
    return output


def optional_array(path: Path, length: int) -> np.ndarray:
    if not path.exists():
        return np.full(length, np.nan, dtype=np.float32)
    values = np.load(path).astype(np.float32)
    if len(values) != length:
        raise ValueError((path, len(values), length))
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--v24-data", type=Path, default=ROOT / "experiments/v24_full_hetero/cloud_dataset")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    train = load_split(args.root / "train", True)
    test = load_split(args.root / "test", False)
    with np.load(args.v24_data / "rootcause_v24_full_hetero.npz", allow_pickle=False) as source:
        base = {name: source[name] for name in source.files}
    if sum(len(item["alarms"]) for item in train) != len(base["train_labels"]):
        raise ValueError("train alarm alignment differs from V24")
    if sum(len(item["alarms"]) for item in test) != len(base["test_v11"]):
        raise ValueError("test alarm alignment differs from V24")

    connected_folds, connected_report = connected_group_folds(train, 5)
    template_folds, template_report = balanced_group_folds(
        [order["signature"] for order in train], 5
    )
    station_folds, station_report = balanced_group_folds(
        [
            ("station", tuple(order["station_ids"]))
            if order["station_ids"]
            else ("no_station", order["signature"])
            for order in train
        ],
        5,
    )
    arrays = {
        "train_alarm_ptr": base["train_alarm_ptr"],
        "test_alarm_ptr": base["test_alarm_ptr"],
        "train_v11": base["train_v11"],
        "test_v11": base["test_v11"],
        "train_labels": base["train_labels"],
        "train_counts": base["train_counts"],
        "train_folds": template_folds,
        "train_station_folds": station_folds,
        "train_connected_folds": connected_folds,
        "train_legacy_folds": base["train_folds"],
        "test_champion_mask": base["test_champion_mask"],
        "train_alarm_x": base["train_alarm_x"],
        "test_alarm_x": base["test_alarm_x"],
        "train_path_node_type": base["train_path_node_type"],
        "test_path_node_type": base["test_path_node_type"],
        "train_path_edge_type": base["train_path_edge_type"],
        "test_path_edge_type": base["test_path_edge_type"],
        "train_path_length": base["train_path_length"],
        "test_path_length": base["test_path_length"],
        "train_v13": 0.25 * optional_array(args.root / "codexgz/v13/v13_oof_context.npy", len(base["train_labels"]))
        + 0.75 * optional_array(args.root / "codexgz/v13/v13_oof_meta.npy", len(base["train_labels"])),
        "test_v13": optional_array(args.root / "codexgz/v13/v13_test_scores.npy", len(base["test_v11"])),
        "train_v19": optional_array(args.root / "experiments/v19_cloud/v19_graph_time_oof_mean.npy", len(base["train_labels"])),
        "test_v19": optional_array(args.root / "experiments/v16/v16_graph_time_test_scores.npy", len(base["test_v11"])),
    }
    np.savez_compressed(args.output / "v25_semantic_router.npz", **arrays)
    with gzip.open(args.output / "semantic_records.json.gz", "wt", encoding="utf-8") as handle:
        json.dump({"train": train, "test": test}, handle, ensure_ascii=False)
    metadata = {
        "version": "v25-semantic-router-1",
        "train_orders": len(train),
        "test_orders": len(test),
        "train_alarms": len(base["train_labels"]),
        "test_alarms": len(base["test_v11"]),
        "positive_labels": int(base["train_labels"].sum()),
        "split": {
            "primary_template": template_report,
            "station_stress": station_report,
            "connected_component_audit": connected_report,
            "connected_component_used_for_training": False
        },
        "excluded_from_model": ["rid", "raw IP", "raw UUID"],
        "unlabeled_test_used_for": ["domain language adaptation"],
    }
    write_json(args.output / "metadata.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
