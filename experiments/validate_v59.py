"""Validate all V59 coded submissions and phase-two invariants."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
OUT = EXP / "v59_extended_coded_campaign"
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
    matrix = np.asarray(report["matrix"], dtype=np.int8)
    assert matrix.shape == (75, 161)
    assert np.linalg.matrix_rank(matrix) == 75 == report["matrix_rank"]
    assert report["candidate_count"] == 161
    assert len(report["probes"]) == 75
    assert report["recovery_diagnostic"]["collisions"] == 0
    assert report["recovery_diagnostic"]["proven_unique_rate"] >= 0.8
    combined = report["combined_v58_v59_oracle_diagnostic"]
    assert combined["p05"] >= 0.95
    assert combined["probability_reach_0_95"] >= 0.99
    actions = report["candidate_actions"]
    orders = {action["order_id"] for action in actions}
    assert len(orders) == 161
    v58_orders = {action["order_id"] for action in read_json(EXP / "v58_coded_campaign/report.json")["candidate_actions"]}
    assert not orders & v58_orders
    for probe_id in report["probe_order"]:
        manifest = report["probes"][probe_id]
        expected_indices = np.flatnonzero(matrix[manifest["row_index"]]).astype(int).tolist()
        assert manifest["candidate_indices"] == expected_indices
        assert sha256(manifest["path"]) == manifest["sha256"]
        assert validate_csv(manifest["path"], valid) == manifest["predictions"]
        assert manifest["predictions"] == 1035 + manifest["additions"] - manifest["deletions"]
        for row in manifest["score_possibilities"]:
            assert row["batch_delta_tp"] == row["correct_count"] - manifest["deletions"]
            expected = 2 * row["tp"] / (TRUE_ROOTS + manifest["predictions"])
            assert abs(row["score"] - expected) <= 0.5e-9 + 1e-12
    state = read_json(OUT / "state.json")
    assert not state["results"]
    assert sha256(state["checkpoint"]["path"]) == state["checkpoint"]["sha256"]
    print(json.dumps({
        "status": "ok", "matrix_shape": list(matrix.shape),
        "candidate_orders": len(orders), "recovery": report["recovery_diagnostic"],
        "combined_oracle": combined, "first_probe": report["probes"]["v59_code_01"]["path"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
