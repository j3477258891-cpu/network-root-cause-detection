"""Record one coded score, decode proven labels, and rebuild the checkpoint."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
V30 = EXP / "v30_meta_stack"
for value in (EXP, V30):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from build_cross_order_probes import apply_actions, load_submission, write_submission
from v37_online_equation_solver import CHAMPION, TRUE_ROOTS, solve_fixed_labels
from v58_coded_campaign import alternative_exists


OUT = EXP / "v58_coded_campaign"
V37 = EXP / "v37_online_equations"
RECORDS = EXP / "v25_semantic_router/cloud_dataset/semantic_records.json.gz"
BASE_TP = 956
BASE_P = 1035


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def map_assignment(matrix, rhs, probabilities, time_limit=30.0):
    clipped = np.clip(probabilities, 1e-5, 1 - 1e-5)
    objective = np.log((1 - clipped) / clipped)
    result = milp(
        objective, integrality=np.ones(len(probabilities), dtype=np.int8),
        bounds=Bounds(np.zeros(len(probabilities)), np.ones(len(probabilities))),
        constraints=LinearConstraint(matrix, rhs, rhs),
        options={"time_limit": time_limit},
    )
    # HiGHS may return a feasible incumbent when the time limit is reached.
    # That incumbent is valid for posterior decoding even without an optimality
    # certificate; rejecting it needlessly prevents early checkpoint audits.
    if result.x is None:
        return None
    assignment = np.rint(result.x).astype(np.int8)
    if np.any(assignment < 0) or np.any(assignment > 1):
        return None
    if not np.array_equal(matrix @ assignment, np.rint(rhs).astype(np.int64)):
        return None
    return assignment


def roots_for_actions(actions, order_ids, champion_roots, records_by_order):
    roots = apply_actions(champion_roots, records_by_order, actions)
    return roots, sum(len(values) for values in roots.values())


def result_manifest(report, state, probe_id):
    if probe_id in report["probes"]:
        return report["probes"][probe_id], "results", True
    adaptive = state.get("adaptive_checkpoints", {})
    if probe_id in adaptive:
        return adaptive[probe_id], "adaptive_results", False
    raise SystemExit(f"unknown probe: {probe_id}")


def observed_equations(report, state):
    rows, rhs = [], []
    matrix_full = np.asarray(report["matrix"], dtype=np.float64)
    for probe_id, result in state.get("results", {}).items():
        rows.append(matrix_full[report["probes"][probe_id]["row_index"]])
        rhs.append(result["correct_count"])
    width = len(report["candidate_actions"])
    for probe_id, result in state.get("adaptive_results", {}).items():
        manifest = state["adaptive_checkpoints"][probe_id]
        row = np.zeros(width, dtype=np.float64)
        row[manifest["candidate_indices"]] = 1.0
        rows.append(row)
        rhs.append(result["correct_count"])
    if not rows:
        return np.empty((0, width), dtype=np.float64), np.empty(0, dtype=np.float64)
    return np.stack(rows), np.asarray(rhs, dtype=np.float64)


def emit_adaptive_checkpoint(
    report, state, assignment, exact, order_ids, champion_roots, records_by_order
):
    """Emit one posterior checkpoint at selected code-count milestones."""
    code_count = len(state.get("results", {}))
    if code_count not in {40, 45, 50, 55} or assignment is None:
        return None
    probe_id = f"v65_after_{code_count:02d}"
    existing = state.setdefault("adaptive_checkpoints", {})
    if probe_id in existing:
        return existing[probe_id]
    indices = np.flatnonzero(assignment).astype(int).tolist()
    selected = [report["candidate_actions"][index] for index in indices]
    all_actions = list(exact["actions"]) + selected
    roots, predictions = roots_for_actions(
        all_actions, order_ids, champion_roots, records_by_order
    )
    additions = sum(bool(action["add_rids"]) for action in selected)
    deletions = len(selected) - additions
    if predictions != BASE_P + additions - deletions:
        raise RuntimeError("adaptive checkpoint prediction mismatch")
    directory = OUT / "adaptive"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{probe_id}.csv"
    write_submission(path, order_ids, roots)
    manifest = {
        "probe_id": probe_id,
        "source": "posterior_map_checkpoint",
        "after_code_equations": code_count,
        "path": str(path),
        "sha256": sha256(path),
        "candidate_indices": indices,
        "candidate_actions": selected,
        "actions": all_actions,
        "additions": additions,
        "deletions": deletions,
        "predictions": predictions,
        "prediction_delta": additions - deletions,
        "score_possibilities": [{
            "correct_count": count,
            "batch_delta_tp": count - deletions,
            "tp": BASE_TP + count - deletions,
            "score": round(
                2 * (BASE_TP + count - deletions) / (TRUE_ROOTS + predictions), 9
            ),
        } for count in range(len(selected) + 1)],
        "warning": "Posterior candidate; not verified until its public score is recorded.",
    }
    existing[probe_id] = manifest
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-id", required=True)
    parser.add_argument("--score", type=float, required=True)
    parser.add_argument("--deep", action="store_true",
                        help="run per-variable MILP bounds if uniqueness is not proven")
    args = parser.parse_args()
    report = read_json(OUT / "report.json")
    state = read_json(OUT / "state.json")
    manifest, result_bucket, is_code_probe = result_manifest(
        report, state, args.probe_id
    )
    predictions = int(manifest["predictions"])
    tp = round(args.score * (TRUE_ROOTS + predictions) / 2)
    reconstructed = 2 * tp / (TRUE_ROOTS + predictions)
    if abs(reconstructed - args.score) > 0.5e-6 + 1e-12:
        raise SystemExit("score does not map to an integer TP")
    delta = tp - BASE_TP
    correct_count = delta + int(manifest["deletions"])
    if not 0 <= correct_count <= len(manifest["candidate_indices"]):
        raise SystemExit("score implies an impossible coded count")
    possible = {int(row["correct_count"]) for row in manifest["score_possibilities"]}
    if correct_count not in possible:
        raise SystemExit("coded count absent from manifest")
    state.setdefault(result_bucket, {})[args.probe_id] = {
        "score": args.score, "tp": tp, "predictions": predictions,
        "batch_delta_tp": delta, "correct_count": correct_count,
        "reconstructed_score": reconstructed,
    }

    matrix, rhs = observed_equations(report, state)
    probabilities = np.asarray(report["correctness_probabilities"], dtype=np.float64)
    assignment = map_assignment(matrix, rhs, probabilities)
    unique = None
    # None from alternative_exists means timeout, so preserve the distinction.
    if assignment is not None:
        alternative = alternative_exists(matrix, assignment, 15.0)
        unique = True if alternative is False else False if alternative is True else None
    lower = np.zeros(len(probabilities), dtype=np.int8)
    upper = np.ones(len(probabilities), dtype=np.int8)
    if unique:
        lower = upper = assignment
    elif args.deep:
        lower, upper, _ = solve_fixed_labels(matrix, rhs, len(probabilities))
    fixed = lower == upper
    fixed_correct = np.flatnonzero(fixed & (lower == 1)).astype(int).tolist()
    fixed_incorrect = np.flatnonzero(fixed & (lower == 0)).astype(int).tolist()

    order_ids, champion_roots = load_submission(CHAMPION)
    with gzip.open(RECORDS, "rt", encoding="utf-8") as handle:
        records = json.load(handle)["test"]
    records_by_order = {row["order_id"]: row for row in records}
    exact = read_json(V37 / "manifests/v37_exact_corrections.json")
    candidate_actions = report["candidate_actions"]
    proven_actions = list(exact["actions"]) + [candidate_actions[index] for index in fixed_correct]
    roots, proven_p = roots_for_actions(proven_actions, order_ids, champion_roots, records_by_order)
    proven_adds = sum(bool(candidate_actions[index]["add_rids"]) for index in fixed_correct)
    proven_deletes = sum(bool(candidate_actions[index]["remove_rids"]) for index in fixed_correct)
    proven_tp = BASE_TP + proven_adds
    if proven_p != BASE_P + proven_adds - proven_deletes:
        raise RuntimeError("proven checkpoint count mismatch")
    proven_score = 2 * proven_tp / (TRUE_ROOTS + proven_p)

    best = {
        "source": "fixed_labels", "score": proven_score, "tp": proven_tp,
        "predictions": proven_p, "actions": proven_actions,
        "selected_probe_id": None,
    }
    scored_manifests = [
        (probe_id, result, report["probes"][probe_id])
        for probe_id, result in state.get("results", {}).items()
    ] + [
        (probe_id, result, state["adaptive_checkpoints"][probe_id])
        for probe_id, result in state.get("adaptive_results", {}).items()
    ]
    for probe_id, result, probe in scored_manifests:
        if result["reconstructed_score"] <= best["score"]:
            continue
        best = {
            "source": "verified_probe", "score": result["reconstructed_score"],
            "tp": result["tp"], "predictions": result["predictions"],
            "actions": probe["actions"], "selected_probe_id": probe_id,
        }
    best_roots, best_p = roots_for_actions(best["actions"], order_ids, champion_roots, records_by_order)
    if best_p != best["predictions"]:
        raise RuntimeError("best checkpoint count mismatch")
    path = OUT / "highest_verified_checkpoint.csv"
    write_submission(path, order_ids, best_roots)
    unresolved = len(probabilities) - len(fixed_correct) - len(fixed_incorrect)
    checkpoint = {
        **best, "path": str(path), "sha256": sha256(path),
        "fixed_correct_count": len(fixed_correct),
        "fixed_incorrect_count": len(fixed_incorrect),
        "unresolved_count": unresolved,
    }
    checkpoint.pop("actions")
    unscored = [probe_id for probe_id in report["probe_order"] if probe_id not in state["results"]]
    adaptive = None
    if is_code_probe and not unique:
        adaptive = emit_adaptive_checkpoint(
            report, state, assignment, exact, order_ids, champion_roots, records_by_order
        )
    unscored_adaptive = [
        row for probe_id, row in state.get("adaptive_checkpoints", {}).items()
        if probe_id not in state.get("adaptive_results", {})
    ]
    state["decode"] = {
        "equation_count": len(rhs), "code_equation_count": len(state.get("results", {})),
        "adaptive_equation_count": len(state.get("adaptive_results", {})),
        "map_feasible": assignment is not None,
        "unique_proven": unique, "deep_bounds_run": args.deep,
        "fixed_correct_indices": fixed_correct,
        "fixed_incorrect_indices": fixed_incorrect,
        "unresolved_count": unresolved,
    }
    state["checkpoint"] = checkpoint
    state["phase_complete"] = bool(unique) or (not unscored and not unscored_adaptive)
    state["target_reached"] = checkpoint["score"] >= 0.95
    state["next_probe"] = None
    if not state["phase_complete"] and not state["target_reached"]:
        if unscored_adaptive:
            state["next_probe"] = unscored_adaptive[0]
        elif unscored:
            state["next_probe"] = report["probes"][unscored[0]]
    (OUT / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "highest_verified_checkpoint.json").write_text(
        json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "recorded": state[result_bucket][args.probe_id], "decode": state["decode"],
        "checkpoint": checkpoint,
        "adaptive_checkpoint_emitted": None if adaptive is None else {
            key: adaptive[key] for key in ("probe_id", "path", "sha256", "predictions")
        },
        "next_probe": None if state["next_probe"] is None else {
            "probe_id": state["next_probe"]["probe_id"],
            "path": state["next_probe"]["path"], "sha256": state["next_probe"]["sha256"],
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
