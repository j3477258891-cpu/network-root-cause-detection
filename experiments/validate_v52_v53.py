"""Validate V52 frontier batches and the V53 adaptive equation package."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
RECORDS = EXPERIMENTS / "v25_semantic_router/cloud_dataset/semantic_records.json.gz"
TRUE_ROOTS = 1044


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_records():
    with gzip.open(RECORDS, "rt", encoding="utf-8") as handle:
        rows = json.load(handle)["test"]
    return {
        row["order_id"]: {alarm["rid"] for alarm in row["alarms"]}
        for row in rows
    }


def validate_submission(path, valid):
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


def validate_manifest(manifest, valid):
    path = Path(manifest["path"])
    assert path.exists()
    assert sha256(path) == manifest["sha256"]
    predictions = validate_submission(path, valid)
    assert predictions == int(manifest["predictions"])
    candidate_actions = [
        action for action in manifest["actions"]
        if action["source"] != "historical_leaderboard_equations"
    ]
    orders = [action["order_id"] for action in candidate_actions]
    assert len(orders) == len(set(orders))
    for row in manifest["score_possibilities"]:
        exact = 2 * int(row["tp"]) / (TRUE_ROOTS + predictions)
        assert abs(float(row["score"]) - exact) <= 0.5e-9 + 1e-12
    return candidate_actions


def main():
    valid = load_records()
    v52 = read_json(EXPERIMENTS / "v52_precision_frontier/report.json")
    all_orders = set()
    all_nodes = set()
    for manifest in v52["probes"].values():
        actions = validate_manifest(manifest, valid)
        for action in actions:
            assert action["order_id"] not in all_orders
            all_orders.add(action["order_id"])
            for rid in action["remove_rids"] + action["add_rids"]:
                key = (action["order_id"], rid)
                assert key not in all_nodes
                all_nodes.add(key)
    assert len(v52["probes"]) == 16
    assert len(all_orders) == 64

    v53 = read_json(EXPERIMENTS / "v53_active_equation/report.json")
    probe_actions = validate_manifest(v53["probe"], valid)
    assert v53["selected"]["query_id"] == "v45_map_prefix_02"
    assert len(probe_actions) == 2
    assert v53["probe"]["predictions"] == 1035
    assert [row["tp"] for row in v53["probe"]["score_possibilities"]] == [955, 956, 957]
    for delta, outcome in v53["outcome_files"].items():
        path = Path(outcome["path"])
        assert path.exists()
        assert sha256(path) == outcome["sha256"]
        predictions = validate_submission(path, valid)
        assert predictions == int(outcome["predictions"])
        exact = 2 * int(outcome["tp"]) / (TRUE_ROOTS + predictions)
        assert abs(exact - float(outcome["score"])) < 1e-12
        assert int(delta) in (-1, 0, 1)
    print(json.dumps({
        "v52_probes": len(v52["probes"]),
        "v52_disjoint_candidate_orders": len(all_orders),
        "v53_probe": v53["probe"]["path"],
        "v53_sha256": v53["probe"]["sha256"],
        "v53_outcomes": v53["outcome_files"],
        "status": "ok",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
