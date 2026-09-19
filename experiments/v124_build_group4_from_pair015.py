"""Build V124 group 04 on top of the newly verified pair-015 champion."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import v124_build_group_probes as v124


ROOT = Path(r"D:\zgyidong")
OUT = ROOT / "experiments/v124_campaign"
BASE = OUT / "v124_single_pair_015_from_probe13.csv"
CATALOG = ROOT / "experiments/v120_swap_campaign/candidate_catalog.json"
GROUP = ["v120_pair_030", "v120_pair_054", "v120_pair_033"]


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({"order_id": row["order_id"], "roots": json.loads(row["output"])["rootcause"]})
    return rows


def main() -> None:
    cat = json.loads(CATALOG.read_text(encoding="utf-8"))
    by_id = {c["candidate_id"]: c for c in cat["candidates"]}
    alarms, _ = v124.v121.load_alarm_records()
    base_rows = load_rows(BASE)
    base_by = {r["order_id"]: {n["@rid"] for n in r["roots"]} for r in base_rows}
    actions = []
    for cid in GROUP:
        c = by_id[cid]
        assert c["remove_rid"] in base_by[c["order_id"]]
        assert c["add_rid"] not in base_by[c["order_id"]]
        actions.append({"order_id": c["order_id"], "remove_rid": c["remove_rid"], "add_rid": c["add_rid"]})
    rows = v124.apply_actions(base_rows, actions, alarms)
    valid, predictions, message = v124.validate(rows, 1035)
    if not valid:
        raise RuntimeError(message)
    path = OUT / "v124_group_04_from_pair015.csv"
    v124.write_csv(path, rows)
    meta = {
        "version": "v124-group4-adaptive",
        "base": str(BASE),
        "base_sha256": v124.sha256(BASE),
        "base_public_f1": 0.921597,
        "base_tp": 958,
        "candidates": GROUP,
        "predictions": predictions,
        "orders": len(rows),
        "sha256": v124.sha256(path),
        "expected_score_branches": {
            "0.918711": {"group_delta": -3},
            "0.919673": {"group_delta": -2},
            "0.920635": {"group_delta": -1},
            "0.921597": {"group_delta": 0},
            "0.922559": {"group_delta": 1},
            "0.923521": {"group_delta": 2},
            "0.924483": {"group_delta": 3}
        }
    }
    v124.write_json(OUT / "v124_group_04_from_pair015.json", meta)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
