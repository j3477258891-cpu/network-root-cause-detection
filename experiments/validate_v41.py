"""Independent structural and hash validation for V41 submissions."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/v41_equation_aware"
REPORT = OUT / "report.json"
RECORDS = ROOT / "experiments/v25_semantic_router/cloud_dataset/semantic_records.json.gz"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    order_ids, roots = [], {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            order_ids.append(row["order_id"])
            roots[row["order_id"]] = json.loads(row["output"])["rootcause"]
    return order_ids, roots


def main():
    report = read_json(REPORT)
    base_ids, base = load(report["champion"]["path"])
    with gzip.open(RECORDS, "rt", encoding="utf-8") as handle:
        records = json.load(handle)["test"]
    valid = {row["order_id"]: {alarm["rid"] for alarm in row["alarms"]} for row in records}
    assert len(base_ids) == len(set(base_ids)) == 546
    assert sum(map(len, base.values())) == report["champion"]["predictions"] == 1035

    results = []
    for name, metadata in report["probes"].items():
        manifest = read_json(OUT / "manifests" / f"{name}.json")
        path = Path(manifest["path"])
        order_ids, roots = load(path)
        assert order_ids == base_ids
        assert sha256(path) == manifest["sha256"] == metadata["sha256"]
        assert sum(map(len, roots.values())) == manifest["predictions"]
        action_orders = [action["order_id"] for action in manifest["actions"]]
        assert len(action_orders) == len(set(action_orders))

        expected_changed = set(action_orders)
        actual_changed = {oid for oid in base_ids if roots[oid] != base[oid]}
        assert actual_changed == expected_changed
        for oid, values in roots.items():
            rids = [node["@rid"] for node in values]
            assert 1 <= len(rids) <= 8
            assert len(rids) == len(set(rids))
            assert set(rids) <= valid[oid]
        for action in manifest["actions"]:
            oid = action["order_id"]
            before = {node["@rid"] for node in base[oid]}
            after = {node["@rid"] for node in roots[oid]}
            assert before - after == set(action["remove_rids"])
            assert after - before == set(action["add_rids"])
        results.append({
            "probe_id": name, "predictions": manifest["predictions"],
            "sha256": manifest["sha256"], "status": "PASS",
        })
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
