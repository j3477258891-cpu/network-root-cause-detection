"""Audit the 20 highest ``seed_min_margin`` V16 test swaps.

The V16 catalog was generated against an older 1,059-root checkpoint.  This
read-only script checks membership against the current 1,035-root champion and
solves each swap's label difference against the 37 public scored equations.
It never writes a submission CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CATALOG = ROOT / "experiments/v16/v16_test_swap_catalog.csv"
BASE = ROOT / "experiments/v60_combined_checkpoint/highest_verified_combined.csv"
sys.path.insert(0, str(HERE))
import audit_real_scores as audit  # noqa: E402


def load_baseline() -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    with BASE.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            result[row["order_id"]] = {
                node["@rid"] for node in json.loads(row["output"])["rootcause"]
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--time-limit", type=float, default=20.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "v16_top20_bounds.json",
    )
    args = parser.parse_args()

    records, _ = audit.collect_records()
    alarms, _ = audit.load_test_universe()
    universe_keys = sorted(alarms)
    universe = {key: index for index, key in enumerate(universe_keys)}
    matrix, rhs, _ = audit.build_equations(records, universe)
    baseline = load_baseline()

    with CATALOG.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: float(row["seed_min_margin"]), reverse=True)
    output_rows: list[dict] = []
    for rank, row in enumerate(rows[: args.top], start=1):
        order_id = row["order_id"]
        remove_rid = row["removed_rid"]
        add_rid = row["added_rid"]
        remove_key = (order_id, remove_rid)
        add_key = (order_id, add_rid)
        in_universe = remove_key in universe and add_key in universe
        if in_universe:
            bounds = audit.solve_min_max(
                {
                    universe[add_key]: 1.0,
                    universe[remove_key]: -1.0,
                },
                matrix,
                rhs,
                args.time_limit,
            )
        else:
            bounds = {
                "min": None,
                "max": None,
                "optimal": False,
                "reason": "missing test-universe variable",
            }
        roots = baseline.get(order_id, set())
        output_rows.append(
            {
                "rank_by_seed_min_margin": rank,
                "order_id": order_id,
                "remove_rid": remove_rid,
                "add_rid": add_rid,
                "seed_min_margin": float(row["seed_min_margin"]),
                "margin": float(row["margin"]),
                "meta_mean": float(row["meta_mean"]),
                "meta_seed_min": float(row["meta_seed_min"]),
                "remove_in_current_baseline": remove_rid in roots,
                "add_in_current_baseline": add_rid in roots,
                "base_root_count": len(roots),
                "valid_current_swap": remove_rid in roots and add_rid not in roots,
                "in_test_universe": in_universe,
                "equation_bounds": bounds,
            }
        )

    payload = {
        "version": 1,
        "catalog": str(CATALOG),
        "catalog_rows": len(rows),
        "top_n": len(output_rows),
        "scored_equation_count": len(records),
        "equation_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "baseline": {
            "path": str(BASE),
            "predictions": sum(len(values) for values in baseline.values()),
            "tp": 956,
        },
        "rows": output_rows,
        "summary": {
            "valid_current_swap_count": sum(row["valid_current_swap"] for row in output_rows),
            "equation_min_positive_count": sum(
                row["equation_bounds"].get("min") == 1
                for row in output_rows
                if row["equation_bounds"].get("optimal")
            ),
            "equation_fixed_negative_count": sum(
                row["equation_bounds"].get("min") == -1
                and row["equation_bounds"].get("max") == -1
                for row in output_rows
                if row["equation_bounds"].get("optimal")
            ),
            "equation_fixed_zero_count": sum(
                row["equation_bounds"].get("min") == 0
                and row["equation_bounds"].get("max") == 0
                for row in output_rows
                if row["equation_bounds"].get("optimal")
            ),
        },
        "interpretation": [
            "seed_min_margin is a model stability score, not a calibrated probability.",
            "A [-1,+1] equation bound means the real-score evidence does not identify the swap direction.",
            "No V124 probe or submission CSV is generated by this audit.",
        ],
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), **payload["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
