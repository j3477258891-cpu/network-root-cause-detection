"""Validate the complete V55 probe package."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
OUT = EXP / "v55_pair_equation"
TRUE_ROOTS = 1044


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_csv(path, valid):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["order_id", "output"]
        rows = list(reader)
    assert len(rows) == 546
    assert len({row["order_id"] for row in rows}) == 546
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
    probe = report["probe"]
    assert probe["predictions"] == 1035
    assert digest(probe["path"]) == probe["sha256"]
    assert validate_csv(probe["path"], valid) == 1035
    candidate = probe["candidate_actions"]
    assert len(candidate) == 2
    assert len({row["order_id"] for row in candidate}) == 2
    assert sum(len(row["add_rids"]) for row in candidate) == 1
    assert sum(len(row["remove_rids"]) for row in candidate) == 1
    assert [row["query_delta_tp"] for row in probe["score_possibilities"]] == [-1, 0, 1]
    assert [row["tp"] for row in probe["score_possibilities"]] == [955, 956, 957]
    for row in probe["score_possibilities"]:
        expected = 2 * row["tp"] / (TRUE_ROOTS + 1035)
        assert abs(row["score"] - expected) <= 0.5e-9 + 1e-12
    for delta, outcome in report["outcome_files"].items():
        assert int(delta) in (-1, 0, 1)
        assert digest(outcome["path"]) == outcome["sha256"]
        predictions = validate_csv(outcome["path"], valid)
        assert predictions == outcome["predictions"]
        expected = 2 * outcome["tp"] / (TRUE_ROOTS + predictions)
        assert abs(outcome["score"] - expected) < 1e-12
        assert outcome["score"] >= 2 * 956 / (TRUE_ROOTS + 1035)
    print(json.dumps({
        "status": "ok", "probe": probe["path"], "sha256": probe["sha256"],
        "predictions": probe["predictions"], "outcomes": report["outcome_files"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
