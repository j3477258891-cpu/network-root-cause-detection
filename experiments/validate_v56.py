"""Validate V56 manifests, submissions, score tables, and disjointness."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
OUT = EXP / "v56_quantitative_campaign"
TRUE_ROOTS = 1044


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_csv(path, valid):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["order_id", "output"]
        rows = list(reader)
    assert len(rows) == 546
    assert {row["order_id"] for row in rows} == set(valid)
    predictions = 0
    for row in rows:
        roots = json.loads(row["output"])["rootcause"]
        assert 1 <= len(roots) <= 8
        rids = [node["@rid"] for node in roots]
        assert len(rids) == len(set(rids))
        assert set(rids) <= valid[row["order_id"]]
        predictions += len(rids)
    return predictions


def main():
    with gzip.open(EXP / "v25_semantic_router/cloud_dataset/semantic_records.json.gz", "rt", encoding="utf-8") as handle:
        records = json.load(handle)["test"]
    valid = {row["order_id"]: {alarm["rid"] for alarm in row["alarms"]} for row in records}
    report = read_json(OUT / "report.json")
    assert len(report["probes"]) == 15
    assert report["candidate_count"] == 61
    assert report["additions"] == 32
    assert report["deletions"] == 29
    assert report["schedule"] == {
        "v49_and_v47": 2, "v56_quantitative_probes": 15,
        "checkpoint": 1, "reserve": 2,
    }
    all_nodes, all_orders = set(), set()
    for probe_id in report["probe_order"]:
        manifest = report["probes"][probe_id]
        assert sha256(manifest["path"]) == manifest["sha256"]
        assert validate_csv(manifest["path"], valid) == manifest["predictions"]
        assert manifest["predictions"] == 1035 + manifest["prediction_delta"]
        assert manifest["candidate_count"] in (4, 5)
        expected_deltas = list(range(-manifest["deletions"], manifest["additions"] + 1))
        actual_deltas = [row["batch_delta_tp"] for row in manifest["score_possibilities"]]
        assert actual_deltas == expected_deltas
        for row in manifest["score_possibilities"]:
            expected = 2 * row["tp"] / (TRUE_ROOTS + manifest["predictions"])
            assert abs(row["score"] - expected) <= 0.5e-9 + 1e-12
        for action in manifest["candidate_actions"]:
            assert action["order_id"] not in all_orders
            all_orders.add(action["order_id"])
            for rid in action["remove_rids"] + action["add_rids"]:
                key = (action["order_id"], rid)
                assert key not in all_nodes
                all_nodes.add(key)
    assert len(all_orders) == len(all_nodes) == 61
    upper = report["optimistic_upper_bound"]
    assert upper["score"] < 0.95
    combined = report["combined_v49_upper_bound"]
    assert combined["score"] >= 0.95
    assert combined["minimum_v49_delta_if_all_v56_correct"] == 2
    state = read_json(OUT / "state.json")
    assert not state["results"]
    assert state["checkpoint"]["tp"] == 956
    assert state["checkpoint"]["predictions"] == 1035
    print(json.dumps({
        "status": "ok", "probes": len(report["probes"]),
        "candidate_orders": len(all_orders),
        "v56_upper": upper, "combined_v49_upper": combined,
        "posterior_diagnostic": report["posterior_diagnostic"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
