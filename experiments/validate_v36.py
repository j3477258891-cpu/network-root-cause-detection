"""Independent CSV/manifest audit for V36 selective count probes."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(r"D:\zgyidong")
BASE = ROOT / "experiments/v30_meta_stack/submissions/v30_cross_order_top5.csv"
OUT = ROOT / "experiments/v36_selective_counts"
RECORDS = ROOT / "experiments/v25_semantic_router/cloud_dataset/semantic_records.json.gz"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    order, roots = [], {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            order.append(row["order_id"])
            roots[row["order_id"]] = {node["@rid"] for node in json.loads(row["output"])["rootcause"]}
    return order, roots


def main():
    report = read_json(OUT / "report.json")
    base_order, base = load(BASE)
    with gzip.open(RECORDS, "rt", encoding="utf-8") as handle:
        records = json.load(handle)["test"]
    legal = {row["order_id"]: {alarm["rid"] for alarm in row["alarms"]} for row in records}
    assert len(base_order) == len(set(base_order)) == 546
    touched = set()
    results = []
    for name, metadata in report["probes"].items():
        manifest = read_json(OUT / "manifests" / f"{name}.json")
        path = Path(metadata["path"])
        order, roots = load(path)
        assert order == base_order
        assert sha256(path) == metadata["sha256"] == manifest["sha256"]
        assert sum(map(len, roots.values())) == metadata["predictions"] == manifest["predictions"]
        assert all(1 <= len(value) <= 8 for value in roots.values())
        action = manifest["actions"][0]
        oid = action["order_id"]
        removed, added = base[oid] - roots[oid], roots[oid] - base[oid]
        assert removed == set(action["remove_rids"])
        assert added == set(action["add_rids"])
        assert removed | added <= legal[oid]
        assert all(roots[key] == base[key] for key in roots if key != oid)
        action_nodes = {(oid, rid) for rid in removed | added}
        assert touched.isdisjoint(action_nodes)
        touched |= action_nodes
        results.append({"probe": name, "predictions": manifest["predictions"],
                        "sha256": manifest["sha256"]})
    assert len(results) == 3
    print(json.dumps({"status": "PASS", "probes": results, "disjoint_nodes": len(touched)}, indent=2))


if __name__ == "__main__":
    main()
