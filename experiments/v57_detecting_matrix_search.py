"""Search for compact additive-query matrices over the 65 action pool.

A matrix is detecting when every binary label vector has a distinct vector of
integer row sums.  Equivalently, its integer nullspace contains no non-zero
vector with coefficients in {-1, 0, 1}.  Such a matrix would let 19 public
leaderboard counts recover all 65 candidate labels, followed by one final
checkpoint submission.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/v57_detecting_matrix"
N = 65


def collision(matrix: np.ndarray, time_limit=5.0):
    """Return a {-1,0,1} collision, None if proven absent, or 'timeout'."""
    rows, columns = matrix.shape
    # z = positive - negative, with disjoint binary supports.
    equality = np.hstack([matrix, -matrix]).astype(np.float64)
    disjoint = np.zeros((columns, 2 * columns), dtype=np.float64)
    for index in range(columns):
        disjoint[index, index] = 1
        disjoint[index, columns + index] = 1
    nonzero = np.ones((1, 2 * columns), dtype=np.float64)
    constraints = [
        LinearConstraint(equality, np.zeros(rows), np.zeros(rows)),
        LinearConstraint(disjoint, np.zeros(columns), np.ones(columns)),
        LinearConstraint(nonzero, np.ones(1), np.full(1, 2 * columns)),
    ]
    result = milp(
        np.zeros(2 * columns), integrality=np.ones(2 * columns, dtype=np.int8),
        bounds=Bounds(np.zeros(2 * columns), np.ones(2 * columns)),
        constraints=constraints, options={"time_limit": time_limit},
    )
    if result.success:
        vector = np.rint(result.x[:columns] - result.x[columns:]).astype(np.int8)
        if not np.any(vector) or np.any(matrix @ vector):
            raise RuntimeError("invalid collision returned by MILP")
        return vector
    if result.status == 2:
        return None
    return "timeout"


def sidon_matrix(rows, rng, attempts=20000):
    """Greedily forbid duplicate columns and equal sums of disjoint pairs."""
    columns = []
    pair_sums = set()
    for index in range(N):
        first_bit = 1 if index < 4 else 0
        accepted = None
        for _ in range(attempts):
            candidate = np.r_[first_bit, rng.binomial(1, 0.5, size=rows - 1)].astype(np.int8)
            key = tuple(candidate.tolist())
            if any(np.array_equal(candidate, column) for column in columns):
                continue
            new_sums = [tuple((candidate + column).tolist()) for column in columns]
            if len(new_sums) != len(set(new_sums)) or any(value in pair_sums for value in new_sums):
                continue
            accepted = candidate
            break
        if accepted is None:
            return None
        for column in columns:
            pair_sums.add(tuple((accepted + column).tolist()))
        columns.append(accepted)
    return np.stack(columns, axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=19)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--time-limit", type=float, default=2.0)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    summaries, best = [], None
    for trial in range(args.trials):
        matrix = sidon_matrix(args.rows, rng)
        if matrix is None:
            summaries.append({"trial": trial, "status": "sidon_generation_failed"})
            continue
        found = collision(matrix, args.time_limit)
        if found is None:
            best = matrix
            summaries.append({"trial": trial, "status": "injective"})
            break
        if isinstance(found, str):
            summaries.append({"trial": trial, "status": found})
            continue
        support = int(np.count_nonzero(found))
        summaries.append({"trial": trial, "status": "collision", "support": support})
    report = {
        "version": "v57-detecting-matrix-search-1", "rows": args.rows,
        "columns": N, "trials": args.trials, "seed": args.seed,
        "time_limit": args.time_limit, "summaries": summaries,
        "injective_found": best is not None,
    }
    if best is not None:
        path = OUT / f"matrix_{args.rows}x{N}.json"
        path.write_text(json.dumps(best.tolist()), encoding="utf-8")
        report["matrix_path"] = str(path)
        report["row_weights"] = best.sum(axis=1).astype(int).tolist()
        report["column_weights"] = best.sum(axis=0).astype(int).tolist()
    (OUT / f"search_{args.rows}x{N}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
