"""Strict structural validation for the generated V24 dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_ptr(name, ptr, expected_items, total):
    assert ptr.dtype == np.int64, (name, ptr.dtype)
    assert len(ptr) == expected_items + 1, (name, len(ptr), expected_items)
    assert ptr[0] == 0 and ptr[-1] == total, (name, ptr[0], ptr[-1], total)
    assert np.all(ptr[1:] >= ptr[:-1]), name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.data_root / "manifest.json"
    metadata_path = args.data_root / "metadata.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    npz_path = args.data_root / "rootcause_v24_full_hetero.npz"
    assert sha256(npz_path) == manifest["files"][npz_path.name]["sha256"]
    assert sha256(metadata_path) == manifest["files"][metadata_path.name]["sha256"]
    with np.load(npz_path) as archive:
        arrays = {name: archive[name] for name in archive.files}

    required = {
        f"{split}_{name}"
        for split in ("train", "test")
        for name in (
            "node_type", "node_numeric", "edge_src", "edge_dst", "edge_type",
            "alarm_node_index", "alarm_feature_index", "path_node_type",
            "path_edge_type", "path_length", "node_ptr", "edge_ptr", "alarm_ptr",
            "alarm_x", "v11",
        )
    } | {"train_labels", "train_counts", "train_folds", "test_champion_mask"}
    missing = sorted(required - set(arrays))
    assert not missing, missing

    report = {"version": manifest["version"], "splits": {}}
    for split, order_count in (("train", 1634), ("test", 546)):
        nodes = arrays[f"{split}_node_type"]
        edges = arrays[f"{split}_edge_type"]
        alarms = arrays[f"{split}_alarm_node_index"]
        node_ptr = arrays[f"{split}_node_ptr"]
        edge_ptr = arrays[f"{split}_edge_ptr"]
        alarm_ptr = arrays[f"{split}_alarm_ptr"]
        validate_ptr(f"{split}_node_ptr", node_ptr, order_count, len(nodes))
        validate_ptr(f"{split}_edge_ptr", edge_ptr, order_count, len(edges))
        validate_ptr(f"{split}_alarm_ptr", alarm_ptr, order_count, len(alarms))
        assert arrays[f"{split}_node_numeric"].shape == (len(nodes), 18)
        assert arrays[f"{split}_alarm_x"].shape == (len(alarms), 252)
        assert arrays[f"{split}_v11"].shape == (len(alarms),)
        assert arrays[f"{split}_path_node_type"].shape == (len(alarms), 12)
        assert arrays[f"{split}_path_edge_type"].shape == (len(alarms), 11)
        assert np.isfinite(arrays[f"{split}_node_numeric"]).all()
        assert np.isfinite(arrays[f"{split}_alarm_x"]).all()
        assert np.isfinite(arrays[f"{split}_v11"]).all()
        assert nodes.min() >= 0 and nodes.max() < 15
        assert edges.min() >= 0 and edges.max() < 7
        assert alarms.min() >= 0 and alarms.max() < len(nodes)
        assert arrays[f"{split}_alarm_feature_index"].tolist() == list(range(len(alarms)))
        assert np.all((arrays[f"{split}_path_length"] >= 1) & (arrays[f"{split}_path_length"] <= 12))
        for order_index in range(order_count):
            ns, ne = int(node_ptr[order_index]), int(node_ptr[order_index + 1])
            es, ee = int(edge_ptr[order_index]), int(edge_ptr[order_index + 1])
            als, ale = int(alarm_ptr[order_index]), int(alarm_ptr[order_index + 1])
            assert np.all((arrays[f"{split}_edge_src"][es:ee] >= ns) & (arrays[f"{split}_edge_src"][es:ee] < ne))
            assert np.all((arrays[f"{split}_edge_dst"][es:ee] >= ns) & (arrays[f"{split}_edge_dst"][es:ee] < ne))
            assert np.all((alarms[als:ale] >= ns) & (alarms[als:ale] < ne))
            assert 1 <= ale - als
        report["splits"][split] = {
            "orders": order_count,
            "nodes": len(nodes),
            "edges": len(edges),
            "alarms": len(alarms),
            "max_nodes_per_order": int(np.diff(node_ptr).max()),
            "max_edges_per_order": int(np.diff(edge_ptr).max()),
            "max_alarms_per_order": int(np.diff(alarm_ptr).max()),
        }

    assert int(arrays["train_labels"].sum()) == 3041
    assert len(arrays["train_counts"]) == 1634
    assert int(arrays["train_counts"].sum()) == 3041
    assert set(arrays["train_folds"].tolist()) == set(range(5))
    assert int(arrays["test_champion_mask"].sum()) == 1059
    for order_index in range(546):
        start = int(arrays["test_alarm_ptr"][order_index])
        stop = int(arrays["test_alarm_ptr"][order_index + 1])
        count = int(arrays["test_champion_mask"][start:stop].sum())
        assert 1 <= count <= 8, (order_index, count)

    output = args.data_root / "validation_report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({**report, "sha256": sha256(output)}, indent=2))


if __name__ == "__main__":
    main()
