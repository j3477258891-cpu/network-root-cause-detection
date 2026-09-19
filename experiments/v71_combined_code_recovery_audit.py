"""Feasibility audit for a single code over all 290 disjoint actions.

This file deliberately emits no submission.  It measures whether dense count
equations can recover the action correctness vector under the existing priors.
Recovery is scored against sampled truth, not merely by whether MILP returned
some feasible assignment.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(r"D:\zgyidong")
if str(ROOT / ".deps") not in sys.path:
    sys.path.insert(0, str(ROOT / ".deps"))

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


EXP = ROOT / "experiments"
OUT = EXP / "v71_combined_code_recovery"
CAMPAIGNS = ("v58_coded_campaign", "v59_extended_coded_campaign")
BASE_TP = 956
BASE_P = 1035
TRUE_ROOTS = 1044
TARGET = 0.95
ROWS = 64
SEED = 20260820


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def action_key(action: dict) -> tuple:
    return action["order_id"], tuple(action["remove_rids"]), tuple(action["add_rids"])


def load_pool() -> tuple[list[dict], np.ndarray]:
    actions, probabilities = [], []
    for campaign in CAMPAIGNS:
        report = read_json(EXP / campaign / "report.json")
        actions.extend(report["candidate_actions"])
        probabilities.extend(report["correctness_probabilities"])
    if len(actions) != 290 or len({action_key(action) for action in actions}) != 290:
        raise RuntimeError("combined candidate pool is not 290 unique actions")
    if len({action["order_id"] for action in actions}) != 290:
        raise RuntimeError("combined candidate pool is not order-disjoint")
    return actions, np.asarray(probabilities, dtype=np.float64)


def matrix_for(probabilities: np.ndarray) -> np.ndarray:
    """Build balanced dense rows and reject weak/duplicate columns."""
    path = OUT / "matrix.npy"
    if path.exists():
        matrix = np.load(path).astype(np.int8)
        if matrix.shape != (ROWS, len(probabilities)):
            raise ValueError(matrix.shape)
        return matrix
    rng = np.random.default_rng(SEED)
    # Each action appears 28--36 times.  This avoids poorly observed columns
    # while keeping every row close to the maximum-entropy half split.
    matrix = np.zeros((ROWS, len(probabilities)), dtype=np.int8)
    for column in range(len(probabilities)):
        weight = int(rng.integers(28, 37))
        matrix[rng.choice(ROWS, size=weight, replace=False), column] = 1
    attempts = 0
    while len({tuple(matrix[:, col]) for col in range(matrix.shape[1])}) != matrix.shape[1]:
        seen = {}
        for column in range(matrix.shape[1]):
            key = tuple(matrix[:, column])
            if key not in seen:
                seen[key] = column
                continue
            weight = int(rng.integers(28, 37))
            matrix[:, column] = 0
            matrix[rng.choice(ROWS, size=weight, replace=False), column] = 1
            attempts += 1
            if attempts > 10000:
                raise RuntimeError("failed to generate unique columns")
    np.save(path, matrix)
    return matrix


def map_assignment(matrix: np.ndarray, rhs: np.ndarray, probabilities: np.ndarray,
                   time_limit: float) -> tuple[np.ndarray | None, dict]:
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    objective = np.log((1 - clipped) / clipped)
    started = time.perf_counter()
    result = milp(
        objective,
        integrality=np.ones(len(probabilities), dtype=np.int8),
        bounds=Bounds(np.zeros(len(probabilities)), np.ones(len(probabilities))),
        constraints=LinearConstraint(matrix, rhs, rhs),
        options={"time_limit": time_limit, "presolve": True},
    )
    elapsed = time.perf_counter() - started
    diagnostic = {
        "status": int(result.status), "success": bool(result.success),
        "message": str(result.message), "elapsed_seconds": elapsed,
        "mip_gap": None if getattr(result, "mip_gap", None) is None else float(result.mip_gap),
    }
    if result.x is None:
        return None, diagnostic
    assignment = np.rint(result.x).astype(np.int8)
    if not np.array_equal(matrix @ assignment, rhs):
        return None, diagnostic
    return assignment, diagnostic


def checkpoint_score(actions: list[dict], selected: np.ndarray, truth: np.ndarray) -> float:
    tp, predictions = BASE_TP, BASE_P
    for index in np.flatnonzero(selected):
        if actions[index]["add_rids"]:
            predictions += 1
            tp += int(truth[index])
        else:
            predictions -= 1
            tp -= 1 - int(truth[index])
    return 2 * tp / (TRUE_ROOTS + predictions)


def entropy_bits(probabilities: np.ndarray) -> float:
    values = np.clip(probabilities, 1e-12, 1 - 1e-12)
    return float(np.sum(-values * np.log2(values) - (1 - values) * np.log2(1 - values)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--prefixes", default="45,50,55,60,64")
    parser.add_argument("--time-limit", type=float, default=45.0)
    args = parser.parse_args()
    prefixes = [int(value) for value in args.prefixes.split(",")]
    if not prefixes or min(prefixes) < 1 or max(prefixes) > ROWS:
        raise SystemExit(f"prefixes must be within 1..{ROWS}")
    OUT.mkdir(parents=True, exist_ok=True)
    actions, probabilities = load_pool()
    matrix = matrix_for(probabilities)
    rng = np.random.default_rng(SEED + 71)
    rows = {prefix: [] for prefix in prefixes}
    for trial in range(args.trials):
        truth = (rng.random(len(probabilities)) < probabilities).astype(np.int8)
        for prefix in prefixes:
            current = matrix[:prefix]
            assignment, solver = map_assignment(
                current, current @ truth, probabilities, args.time_limit
            )
            row = {"trial": trial, "solver": solver, "solved": assignment is not None}
            if assignment is not None:
                error = float(np.mean(assignment != truth))
                score = checkpoint_score(actions, assignment, truth)
                oracle = checkpoint_score(actions, truth, truth)
                row.update({
                    "label_error": error, "exact_recovery": bool(error == 0),
                    "checkpoint_score": score, "oracle_score": oracle,
                    "score_regret": oracle - score, "target_reached": bool(score >= TARGET),
                })
            rows[prefix].append(row)
            print(json.dumps({"prefix": prefix, **row}, ensure_ascii=False), flush=True)
    summary = {}
    for prefix, values in rows.items():
        solved = [row for row in values if row["solved"]]
        summary[str(prefix)] = {
            "trials": len(values), "solved": len(solved),
            "exact_recoveries": sum(row.get("exact_recovery", False) for row in solved),
            "mean_label_error": None if not solved else float(np.mean([row["label_error"] for row in solved])),
            "mean_checkpoint_score": None if not solved else float(np.mean([row["checkpoint_score"] for row in solved])),
            "minimum_checkpoint_score": None if not solved else float(min(row["checkpoint_score"] for row in solved)),
            "target_reached_count": sum(row.get("target_reached", False) for row in solved),
            "mean_elapsed_seconds": None if not solved else float(np.mean([row["solver"]["elapsed_seconds"] for row in solved])),
        }
    report = {
        "version": "v71-combined-code-recovery-audit-1",
        "candidate_count": len(actions), "rows": ROWS,
        "prior_entropy_bits": entropy_bits(probabilities),
        "matrix_rank": int(np.linalg.matrix_rank(matrix)),
        "row_weight_range": [int(matrix.sum(axis=1).min()), int(matrix.sum(axis=1).max())],
        "column_weight_range": [int(matrix.sum(axis=0).min()), int(matrix.sum(axis=0).max())],
        "settings": {"trials": args.trials, "prefixes": prefixes, "time_limit": args.time_limit},
        "summary": summary, "trials": {str(key): value for key, value in rows.items()},
        "gate": {
            "require_all_trials_solved": True,
            "require_all_trials_exact": True,
            "require_all_trial_checkpoints_at_least_0_95": True,
            "passed": all(
                entry["solved"] == entry["trials"]
                and entry["exact_recoveries"] == entry["trials"]
                and entry["target_reached_count"] == entry["trials"]
                for entry in summary.values()
            ),
        },
        "warning": "Prior-sampled recovery audit; still not public leaderboard evidence.",
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "gate": report["gate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
