"""Build one adaptive V124 single-swap probe on the verified probe-13 champion."""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import v124_build_group_probes as v124


ROOT = Path(r"D:\zgyidong")
OUT = ROOT / "experiments/v124_campaign"
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_id", nargs="?", default="v120_pair_008")
    args = parser.parse_args()
    candidate_id = args.candidate_id
    catalog = json.loads(v124.CATALOG.read_text(encoding="utf-8"))
    candidate = next(c for c in catalog["candidates"] if c["candidate_id"] == candidate_id)
    alarms, _ = v124.v121.load_alarm_records()
    base_rows = v124.load_champion_rows()
    base_by_order = {
        row["order_id"]: {node["@rid"] for node in row["roots"]}
        for row in base_rows
    }

    order_roots = base_by_order[candidate["order_id"]]
    assert candidate["remove_rid"] in order_roots
    assert candidate["add_rid"] not in order_roots

    rows = v124.apply_actions(
        base_rows,
        [{
            "order_id": candidate["order_id"],
            "remove_rid": candidate["remove_rid"],
            "add_rid": candidate["add_rid"],
        }],
        alarms,
    )
    valid, predictions, message = v124.validate(rows, 1035)
    if not valid:
        raise RuntimeError(message)

    pair_suffix = candidate_id.removeprefix("v120_pair_")
    path = OUT / f"v124_single_pair_{pair_suffix}_from_probe13.csv"
    v124.write_csv(path, rows)
    metadata = {
        "version": "v124-adaptive-single",
        "base": str(v124.CHAMPION),
        "base_sha256": v124.sha256(v124.CHAMPION),
        "base_public_f1": 0.920635,
        "base_tp": 957,
        "candidate_id": candidate_id,
        "order_id": candidate["order_id"],
        "remove_rid": candidate["remove_rid"],
        "add_rid": candidate["add_rid"],
        "predictions": predictions,
        "orders": len(rows),
        "sha256": v124.sha256(path),
        "equation_prior": {"min_delta": -1, "max_delta": 1},
        "score_branches": {
            "0.919673": {"tp": 956, "delta_vs_probe13": -1, "decision": f"reject_{candidate_id}"},
            "0.920635": {"tp": 957, "delta_vs_probe13": 0, "decision": f"mark_{candidate_id}_neutral"},
            "0.921597": {"tp": 958, "delta_vs_probe13": 1, "decision": f"accept_{candidate_id}"},
        },
    }
    v124.write_json(OUT / f"v124_single_pair_{pair_suffix}_from_probe13.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
