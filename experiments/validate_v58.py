"""Validate the V58 coded submissions and their exact count equations."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
OUT = EXP / "v58_coded_campaign"
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
    assert matrix.shape == (60, 129)
    assert np.linalg.matrix_rank(matrix) == 60 == report["matrix_rank"]
    assert report["candidate_count"] == 129
    assert report["sources"] == {"v49": 4, "v50": 61, "v52": 64}
    assert report["estimated_days_at_two_submissions"] == 31
    assert len(report["probes"]) == 60
    assert report["recovery_diagnostic"]["collisions"] == 0
    assert report["recovery_diagnostic"]["proven_unique_rate"] >= 0.8
    assert report["oracle_diagnostic"]["median"] >= 0.95
    actions = report["candidate_actions"]
    assert len({action["order_id"] for action in actions}) == 129
    nodes = {(action["order_id"], rid) for action in actions
             for rid in action["remove_rids"] + action["add_rids"]}
    assert len(nodes) == 129
    for probe_id in report["probe_order"]:
        manifest = report["probes"][probe_id]
        index = manifest["row_index"]
        expected_indices = np.flatnonzero(matrix[index]).astype(int).tolist()
        assert manifest["candidate_indices"] == expected_indices
        assert sha256(manifest["path"]) == manifest["sha256"]
        assert validate_csv(manifest["path"], valid) == manifest["predictions"]
        assert manifest["predictions"] == 1035 + manifest["additions"] - manifest["deletions"]
        assert len(manifest["score_possibilities"]) == len(expected_indices) + 1
        for row in manifest["score_possibilities"]:
            assert row["batch_delta_tp"] == row["correct_count"] - manifest["deletions"]
            expected = 2 * row["tp"] / (TRUE_ROOTS + manifest["predictions"])
            assert abs(row["score"] - expected) <= 0.5e-9 + 1e-12
    v49 = read_json(EXP / "v49_extended_count_batch/report.json")["manifest"]
    assert report["probes"]["v58_code_01"]["sha256"] == v49["sha256"]
    state = read_json(OUT / "state.json")
    assert not state["results"]
    assert state["checkpoint"]["tp"] == 956
    assert sha256(state["checkpoint"]["path"]) == state["checkpoint"]["sha256"]
    assert validate_csv(state["checkpoint"]["path"], valid) == 1035
    print(json.dumps({
        "status": "ok", "matrix_shape": list(matrix.shape),
        "candidate_orders": len(nodes), "first_probe": report["probes"]["v58_code_01"]["path"],
        "first_sha256": report["probes"]["v58_code_01"]["sha256"],
        "recovery_diagnostic": report["recovery_diagnostic"],
        "oracle_diagnostic": report["oracle_diagnostic"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
