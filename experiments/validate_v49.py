"""Independent audit for V49 and all sixteen exact-base subsets."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/v49_extended_count_batch"
REPORT = OUT / "report.json"
RECORDS = ROOT / "experiments/v25_semantic_router/cloud_dataset/semantic_records.json.gz"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    ids, roots = [], {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            ids.append(row["order_id"])
            roots[row["order_id"]] = json.loads(row["output"])["rootcause"]
    return ids, roots


def main():
    report = read_json(REPORT)
    manifest = report["manifest"]
    base_ids, base = load(manifest["baseline"])
    ids, roots = load(manifest["path"])
    assert ids == base_ids and len(ids) == len(set(ids)) == 546
    assert sha256(manifest["path"]) == manifest["sha256"]
    assert sum(map(len, roots.values())) == manifest["predictions"] == 1037
    assert len(manifest["actions"]) == 5
    action_orders = [row["order_id"] for row in manifest["actions"]]
    assert len(action_orders) == len(set(action_orders))
    with gzip.open(RECORDS, "rt", encoding="utf-8") as handle:
        records = json.load(handle)["test"]
    valid = {row["order_id"]: {alarm["rid"] for alarm in row["alarms"]} for row in records}
    changed = {oid for oid in ids if roots[oid] != base[oid]}
    assert changed == set(action_orders)
    for oid, values in roots.items():
        rids = [node["@rid"] for node in values]
        assert 1 <= len(rids) <= 8 and len(rids) == len(set(rids))
        assert set(rids) <= valid[oid]
    subsets = manifest["subset_manifests"]
    assert len(subsets) == 16
    action_sets = set()
    for name, metadata in subsets.items():
        subset_manifest = read_json(OUT / "manifests" / f"{name}.json")
        subset_ids, subset_roots = load(metadata["path"])
        assert subset_ids == base_ids
        assert sha256(metadata["path"]) == metadata["sha256"] == subset_manifest["sha256"]
        assert sum(map(len, subset_roots.values())) == metadata["predictions"]
        action_set = tuple(sorted(metadata["candidate_action_ids"]))
        assert action_set not in action_sets
        action_sets.add(action_set)
    exact = read_json(ROOT / "experiments/v37_online_equations/manifests/v37_exact_corrections.json")
    assert subsets["v49_exact_only"]["sha256"] == exact["sha256"]
    v47 = read_json(ROOT / "experiments/v47_v36_count_batch/report.json")["manifest"]
    assert report["decision_tree"]["nested_second_submission"] == v47["path"]
    distinct = sorted({
        (row["candidate_delta_tp"], row["tp"], row["score"], row["decision"])
        for row in manifest["score_possibilities"]
    })
    assert len(distinct) == 5
    print(json.dumps({
        "status": "PASS", "path": manifest["path"], "predictions": manifest["predictions"],
        "sha256": manifest["sha256"], "validated_subsets": len(subsets),
        "distinct_outcomes": distinct,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
