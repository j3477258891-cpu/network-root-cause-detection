"""Audit early MAP checkpoints from prefixes of the V58 count code."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
for value in (EXP, EXP / "v30_meta_stack"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from record_v58_result import map_assignment
from v37_online_equation_solver import TRUE_ROOTS


BASE_TP = 956
BASE_P = 1035


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checkpoint_score(actions, selected, truth):
    tp_delta = 0
    prediction_delta = 0
    for index in np.flatnonzero(selected):
        if actions[index]["add_rids"]:
            prediction_delta += 1
            tp_delta += int(truth[index])
        else:
            prediction_delta -= 1
            tp_delta += int(truth[index]) - 1
    return 2 * (BASE_TP + tp_delta) / (TRUE_ROOTS + BASE_P + prediction_delta)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--prefixes", default="5,10,15,20,25,30,35,40,45,50,55,60")
    args = parser.parse_args()
    report = read_json(EXP / "v58_coded_campaign/report.json")
    matrix = np.asarray(report["matrix"], dtype=np.float64)
    probabilities = np.asarray(report["correctness_probabilities"], dtype=np.float64)
    actions = report["candidate_actions"]
    prefixes = [int(value) for value in args.prefixes.split(",")]
    if not prefixes or min(prefixes) < 1 or max(prefixes) > len(matrix):
        raise SystemExit("prefixes must be between 1 and the V58 row count")
    rng = np.random.default_rng(args.seed)
    rows = {prefix: [] for prefix in prefixes}
    failures = {prefix: 0 for prefix in prefixes}
    hamming = {prefix: [] for prefix in prefixes}
    for _ in range(args.trials):
        truth = (rng.random(len(probabilities)) < probabilities).astype(np.int8)
        for prefix in prefixes:
            current = matrix[:prefix]
            assignment = map_assignment(
                current, current @ truth, probabilities, time_limit=args.time_limit
            )
            if assignment is None:
                failures[prefix] += 1
                continue
            rows[prefix].append(checkpoint_score(actions, assignment, truth))
            hamming[prefix].append(float(np.mean(assignment != truth)))
    summary = {}
    for prefix in prefixes:
        values = np.asarray(rows[prefix], dtype=np.float64)
        errors = np.asarray(hamming[prefix], dtype=np.float64)
        summary[str(prefix)] = {
            "solved": len(values),
            "failures": failures[prefix],
            "mean": None if not len(values) else float(values.mean()),
            "p05": None if not len(values) else float(np.quantile(values, 0.05)),
            "median": None if not len(values) else float(np.median(values)),
            "probability_reach_0_95": None if not len(values) else float(np.mean(values >= 0.95)),
            "mean_label_error": None if not len(errors) else float(errors.mean()),
        }
    output = {
        "version": "v65-posterior-checkpoint-audit-1",
        "trials": args.trials,
        "time_limit": args.time_limit,
        "summary": summary,
        "warning": "Prior-sampled simulation, not leaderboard evidence.",
    }
    out = EXP / "v65_posterior_checkpoint_audit"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
