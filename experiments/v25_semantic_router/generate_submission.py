"""Generate V25 CSV artifacts only after the complete final gate passes."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from actions import ACTION_NAMES
from v25_common import Bundle, read_json, validate_submission, write_json


def write_submission(path, bundle, mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["order_id", "output"])
        writer.writeheader()
        ptr = bundle.ptr("test")
        for index, order in enumerate(bundle.records["test"]):
            start, stop = map(int, ptr[index : index + 2])
            roots = []
            for local in np.flatnonzero(mask[start:stop]):
                alarm = order["alarms"][int(local)]
                roots.append({"@rid": alarm["rid"], **alarm["source"]})
            writer.writerow(
                {"order_id": order["order_id"], "output": json.dumps({"rootcause": roots}, ensure_ascii=False)}
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report_path = args.results / "v25_probe_report.json"
    report = read_json(report_path)
    if report.get("mode") != "final" or report.get("status") != "gate_passed":
        raise SystemExit("final gate did not pass; refusing to generate a submission")
    bundle = Bundle.load(args.data_root)
    masks = np.load(args.results / "v25_final_masks.npz")
    main_path = args.output / "v25_semantic_router_p1059.csv"
    rank_path = args.output / "v25_semantic_router_rank_only_p1059.csv"
    write_submission(main_path, bundle, masks["test_mask"].astype(bool))
    write_submission(rank_path, bundle, masks["rank_only_mask"].astype(bool))
    main_validation = validate_submission(main_path, bundle.records["test"], 1059)
    rank_validation = validate_submission(rank_path, bundle.records["test"], 1059)
    action_names = Counter(
        ACTION_NAMES[int(value)] for value in masks["chosen_name"] if ACTION_NAMES[int(value)] != "keep"
    )
    artifact = {
        "main": {"path": str(main_path), **main_validation},
        "rank_only": {"path": str(rank_path), **rank_validation},
        "action_distribution": dict(action_names),
        "gate_report": str(report_path),
    }
    write_json(args.output / "v25_submission_report.json", artifact)
    report["submission_generated"] = True
    report["submission"] = artifact["main"]
    write_json(report_path, report)
    (main_path.with_suffix(".sha256")).write_text(main_validation["sha256"] + "\n", encoding="ascii")
    print(json.dumps(artifact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
