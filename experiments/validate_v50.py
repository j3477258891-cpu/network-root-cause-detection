"""Independent structural and mathematical audit of all V50 batches."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/v50_disjoint_batches"
REPORT = OUT / "report.json"
RECORDS = ROOT / "experiments/v25_semantic_router/cloud_dataset/semantic_records.json.gz"
TRUE_ROOTS = 1044


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
    probes = report["probes"]
    assert len(probes) == len(report["batch_plan"]) == 8
    base_path = next(iter(probes.values()))["baseline"]
    base_ids, base = load(base_path)
    assert len(base_ids) == len(set(base_ids)) == 546
    with gzip.open(RECORDS, "rt", encoding="utf-8") as handle:
        records = json.load(handle)["test"]
    valid = {row["order_id"]: {alarm["rid"] for alarm in row["alarms"]} for row in records}
    all_orders, all_nodes, results = set(), set(), []
    for name, manifest in probes.items():
        ids, roots = load(manifest["path"])
        assert ids == base_ids
        assert sha256(manifest["path"]) == manifest["sha256"]
        assert sum(map(len, roots.values())) == manifest["predictions"]
        candidate_actions = [row for row in manifest["actions"] if row["source"] == "v50_empirical_bayes_stratum"]
        assert len(candidate_actions) == manifest["candidate_count"]
        local_orders = {row["order_id"] for row in candidate_actions}
        assert len(local_orders) == len(candidate_actions)
        assert all_orders.isdisjoint(local_orders)
        all_orders |= local_orders
        changed = {oid for oid in ids if roots[oid] != base[oid]}
        assert changed == {row["order_id"] for row in manifest["actions"]}
        for action in candidate_actions:
            oid = action["order_id"]
            nodes = {(oid, rid) for rid in action["add_rids"] + action["remove_rids"]}
            assert all_nodes.isdisjoint(nodes)
            all_nodes |= nodes
        for oid, values in roots.items():
            rids = [node["@rid"] for node in values]
            assert 1 <= len(rids) <= 8 and len(rids) == len(set(rids))
            assert set(rids) <= valid[oid]
        for possibility in manifest["score_possibilities"]:
            expected = 2 * possibility["tp"] / (TRUE_ROOTS + manifest["predictions"])
            assert abs(expected - possibility["score"]) < 0.5e-9 + 1e-12
        results.append({
            "probe_id": name, "kind": manifest["kind"],
            "candidate_count": manifest["candidate_count"],
            "predictions": manifest["predictions"], "sha256": manifest["sha256"],
        })
    upper = report["pool_upper_bound"]
    assert upper["v50_additions"] == 34 and upper["v50_deletions"] == 29
    assert abs(2 * upper["tp"] / (TRUE_ROOTS + upper["predictions"]) - upper["score"]) < 1e-12
    assert upper["reaches_0_95"] is True and upper["score"] >= 0.95
    print(json.dumps({
        "status": "PASS", "batch_count": len(results),
        "disjoint_candidate_orders": len(all_orders),
        "batches": results, "pool_upper_bound": upper,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
