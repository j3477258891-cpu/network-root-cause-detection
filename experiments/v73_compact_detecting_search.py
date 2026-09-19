"""Search for a proven compact detecting basis for 30-action blocks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(r"D:\zgyidong")
EXP = ROOT / "experiments"
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import numpy as np

from v57_detecting_matrix_search import collision
from v58_coded_campaign import sidon_matrix


OUT = EXP / "v73_compact_detecting"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--columns", type=int, default=30)
    parser.add_argument("--start-seed", type=int, default=1000)
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--time-limit", type=float, default=5.0)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    attempts = []
    found = None
    for offset in range(args.trials):
        seed = args.start_seed + offset
        matrix = sidon_matrix(args.rows, args.columns, args.columns // 2, seed)
        result = collision(matrix, args.time_limit)
        if result is None:
            status = "injective"
            found = matrix
        elif isinstance(result, str):
            status = result
        else:
            status = f"collision_support_{int(np.count_nonzero(result))}"
        attempts.append({"seed": seed, "status": status})
        if offset % 25 == 0 or found is not None:
            print(json.dumps(attempts[-1]), flush=True)
        if found is not None:
            break
    report = {
        "version": "v73-compact-detecting-search-1",
        "rows": args.rows, "columns": args.columns,
        "start_seed": args.start_seed, "trials_requested": args.trials,
        "time_limit": args.time_limit, "attempts": attempts,
        "injective_found": found is not None,
    }
    if found is not None:
        path = OUT / f"basis_{args.rows}x{args.columns}.json"
        path.write_text(json.dumps(found.astype(int).tolist()), encoding="utf-8")
        report.update({
            "basis_path": str(path),
            "row_weights": found.sum(axis=1).astype(int).tolist(),
            "column_weights": found.sum(axis=0).astype(int).tolist(),
            "proof": "MILP proved no non-zero {-1,0,1} null vector.",
        })
    report_path = OUT / f"search_{args.rows}x{args.columns}_seed_{args.start_seed}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "rows", "columns", "trials_requested", "injective_found"
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
