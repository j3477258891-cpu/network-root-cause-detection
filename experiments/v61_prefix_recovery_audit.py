"""Estimate how early a coded campaign can stop with proven unique labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

from v58_coded_campaign import alternative_exists


SPECS = {
    "v58": (EXP / "v58_coded_campaign/report.json", [40, 45, 50, 55, 60]),
    "v59": (EXP / "v59_extended_coded_campaign/report.json", [50, 55, 60, 65, 70, 75]),
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", choices=sorted(SPECS), required=True)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--time-limit", type=float, default=5.0)
    args = parser.parse_args()
    path, prefixes = SPECS[args.campaign]
    report = read_json(path)
    matrix = np.asarray(report["matrix"], dtype=np.int8)
    probabilities = np.asarray(report["correctness_probabilities"], dtype=np.float64)
    rng = np.random.default_rng(20260830 if args.campaign == "v58" else 20260831)
    rows = []
    for trial in range(args.trials):
        labels = (rng.random(len(probabilities)) < probabilities).astype(np.int8)
        first_unique = None
        outcomes = {}
        for prefix in prefixes:
            result = alternative_exists(matrix[:prefix], labels, args.time_limit)
            status = "collision" if result is True else "unique" if result is False else "timeout"
            outcomes[str(prefix)] = status
            if status == "unique":
                first_unique = prefix
                break
        rows.append({"trial": trial, "first_unique_prefix": first_unique, "outcomes": outcomes})
    distribution = {
        str(prefix): sum(row["first_unique_prefix"] == prefix for row in rows)
        for prefix in prefixes
    }
    unresolved = sum(row["first_unique_prefix"] is None for row in rows)
    report_out = {
        "version": "v61-prefix-recovery-audit-1", "campaign": args.campaign,
        "trials": args.trials, "time_limit": args.time_limit,
        "prefixes": prefixes, "first_unique_distribution": distribution,
        "unresolved": unresolved, "rows": rows,
    }
    out = EXP / "v61_prefix_recovery"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.campaign}.json").write_text(
        json.dumps(report_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report_out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
