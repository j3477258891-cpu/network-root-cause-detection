"""Plan post-V37 probes using all exact leaderboard equations.

The planner never treats a model score as an online label.  It separates
historically constrained nodes from fresh nodes, computes exact feasible TP
deltas for every candidate, and emits probes from the original verified
champion with the V37 exact correction included.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
V30 = EXPERIMENTS / "v30_meta_stack"
sys.path.insert(0, str(V30))
sys.path.insert(0, str(EXPERIMENTS))

from build_cross_order_probes import apply_actions, load_submission, write_submission  # noqa: E402
from v37_online_equation_solver import (  # noqa: E402
    CHAMPION,
    CHAMPION_TP,
    TRUE_ROOTS,
    build_system,
    collect_scored_submissions,
)


OUT = EXPERIMENTS / "v41_equation_aware"
RECORDS = EXPERIMENTS / "v25_semantic_router" / "cloud_dataset" / "semantic_records.json.gz"
V36_REPORT = EXPERIMENTS / "v36_selective_counts" / "report.json"
V37_MANIFEST = EXPERIMENTS / "v37_online_equations" / "manifests" / "v37_exact_corrections.json"
V40_REPORT = EXPERIMENTS / "v40_one_sided_boundary" / "report.json"
BASE_TP_AFTER_EXACT = CHAMPION_TP + 1
TARGET_SCORE = 0.95


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def champion_nodes(roots):
    return {(oid, node["@rid"]) for oid, values in roots.items() for node in values}


def action_signature(actions):
    return tuple(sorted(
        (action["order_id"], tuple(sorted(action.get("remove_rids", []))),
         tuple(sorted(action.get("add_rids", []))))
        for action in actions
    ))


def action_coefficients(actions):
    output = defaultdict(int)
    for action in actions:
        oid = action["order_id"]
        for rid in action.get("add_rids", []):
            output[(oid, rid)] += 1
        for rid in action.get("remove_rids", []):
            output[(oid, rid)] -= 1
    return {key: value for key, value in output.items() if value}


def optimize_history(matrix, rhs, objective):
    if not len(objective):
        return 0, 0
    constraints = LinearConstraint(matrix, rhs, rhs)
    bounds = Bounds(np.zeros(len(objective)), np.ones(len(objective)))
    integrality = np.ones(len(objective), dtype=np.int8)
    lo = milp(objective, integrality=integrality, bounds=bounds,
              constraints=constraints, options={"time_limit": 30})
    hi = milp(-objective, integrality=integrality, bounds=bounds,
              constraints=constraints, options={"time_limit": 30})
    if not lo.success or not hi.success:
        raise RuntimeError("failed to optimize a candidate over historical equations")
    return int(round(lo.fun)), int(round(-hi.fun))


def feasible_history_values(matrix, rhs, objective, low, high):
    if not len(objective):
        return {0}
    bounds = Bounds(np.zeros(len(objective)), np.ones(len(objective)))
    integrality = np.ones(len(objective), dtype=np.int8)
    output = set()
    for value in range(low, high + 1):
        local_matrix = np.vstack([matrix, objective])
        local_rhs = np.r_[rhs, value]
        result = milp(
            np.zeros(len(objective)), integrality=integrality, bounds=bounds,
            constraints=LinearConstraint(local_matrix, local_rhs, local_rhs),
            options={"time_limit": 10},
        )
        if result.success:
            output.add(value)
    return output


def fresh_values(coefficients):
    values = {0}
    for coefficient in coefficients:
        values = {current + coefficient * label for current in values for label in (0, 1)}
    return values


def annotate_candidate(candidate, keys, matrix, rhs, base_predictions):
    key_index = {key: index for index, key in enumerate(keys)}
    coefficients = action_coefficients(candidate["actions"])
    history_vector = np.zeros(len(keys), dtype=np.float64)
    fresh = []
    for key, coefficient in coefficients.items():
        if key in key_index:
            history_vector[key_index[key]] += coefficient
        else:
            fresh.append(coefficient)

    history_low, history_high = optimize_history(matrix, rhs, history_vector)
    history_possible = feasible_history_values(
        matrix, rhs, history_vector, history_low, history_high
    )
    fresh_possible = fresh_values(fresh)
    possible = sorted({left + right for left in history_possible for right in fresh_possible})
    prediction_delta = sum(coefficients.values())
    predictions = base_predictions + prediction_delta
    scores = {
        str(delta): 2 * (BASE_TP_AFTER_EXACT + delta) / (TRUE_ROOTS + predictions)
        for delta in possible
    }
    current_score = 2 * BASE_TP_AFTER_EXACT / (TRUE_ROOTS + base_predictions)
    history_rank_gain = 0
    if np.any(history_vector):
        history_rank_gain = int(
            np.linalg.matrix_rank(np.vstack([matrix, history_vector]))
            - np.linalg.matrix_rank(matrix)
        )
    return {
        **candidate,
        "history_overlap": int(np.count_nonzero(history_vector)),
        "fresh_node_count": len(fresh),
        "touched_node_count": len(coefficients),
        "prediction_delta": prediction_delta,
        "feasible_delta_tp": possible,
        "delta_tp_min": min(possible),
        "delta_tp_max": max(possible),
        "score_min": min(scores.values()),
        "score_max": max(scores.values()),
        "accepting_delta_tp": [int(delta) for delta, score in scores.items() if score > current_score],
        "history_rank_gain": history_rank_gain,
        "maximum_information_bits": math.log2(len(possible)) if possible else 0.0,
    }


def collect_candidates():
    candidates = []
    v36 = read_json(V36_REPORT)
    for index, row in enumerate(v36["accepted_candidates"], 1):
        action = {
            "action_id": f"v41_v36_{row['kind']}_{index:03d}",
            "order_id": row["order_id"],
            "remove_rids": [row["rid"]] if row["kind"] == "delete" else [],
            "add_rids": [row["rid"]] if row["kind"] == "add" else [],
            "source": "v36_selective_count_consensus",
            "expected_gain": row["mean_margin"],
            "evidence": row,
        }
        candidates.append({
            "candidate_id": f"v36_{row['kind']}_{index}", "kind": row["kind"],
            "source": "v36", "source_rank": index, "actions": [action],
            "priority": 1000.0 + float(row["mean_margin"]),
        })

    v40 = read_json(V40_REPORT)
    for kind in ("add", "delete"):
        for index, row in enumerate(v40["test_candidates"][kind], 1):
            action = {
                "action_id": f"v41_v40_{kind}_{index:03d}",
                "order_id": row["order_id"],
                "remove_rids": [row["rid"]] if kind == "delete" else [],
                "add_rids": [row["rid"]] if kind == "add" else [],
                "source": "v40_one_sided_rank_average",
                "expected_gain": None,
                "evidence": {**row, "source_rank": index},
            }
            candidates.append({
                "candidate_id": f"v40_{kind}_{index}", "kind": kind,
                "source": "v40", "source_rank": index, "actions": [action],
                "priority": 500.0 - index,
            })

    for name in ("v30_safe_split_a1", "v30_safe_split_a1_complement"):
        manifest = read_json(V30 / "manifests" / f"{name}.json")
        candidates.append({
            "candidate_id": name, "kind": "swap", "source": "v30",
            "source_rank": 1, "actions": manifest["subset_actions"],
            "priority": 750.0,
        })

    deduplicated = {}
    for candidate in candidates:
        signature = action_signature(candidate["actions"])
        current = deduplicated.get(signature)
        if current is None or candidate["priority"] > current["priority"]:
            if current is not None:
                candidate["aliases"] = current.get("aliases", []) + [current["candidate_id"]]
            deduplicated[signature] = candidate
        elif current is not None:
            current.setdefault("aliases", []).append(candidate["candidate_id"])
    return list(deduplicated.values())


def combine_candidate_actions(candidates, exact_actions):
    by_order = {action["order_id"]: dict(action) for action in exact_actions}
    selected = []
    for candidate in candidates:
        orders = {action["order_id"] for action in candidate["actions"]}
        if orders & by_order.keys():
            continue
        for action in candidate["actions"]:
            by_order[action["order_id"]] = dict(action)
        selected.append(candidate)
    return list(by_order.values()), selected


def emit_probe(name, candidates, exact_manifest, order_ids, roots, records_by_order, base_predictions):
    actions, selected = combine_candidate_actions(candidates, exact_manifest["actions"])
    output = apply_actions(roots, records_by_order, actions)
    predictions = sum(map(len, output.values()))
    path = OUT / "submissions" / f"{name}.csv"
    write_submission(path, order_ids, output)

    coefficients = action_coefficients([
        action for candidate in selected for action in candidate["actions"]
    ])
    additions = sum(1 for value in coefficients.values() if value > 0)
    removals = sum(1 for value in coefficients.values() if value < 0)
    possibilities = []
    for delta in range(-removals, additions + 1):
        possibilities.append({
            "candidate_delta_tp": delta,
            "tp": BASE_TP_AFTER_EXACT + delta,
            "score": round(2 * (BASE_TP_AFTER_EXACT + delta) / (TRUE_ROOTS + predictions), 9),
        })
    manifest = {
        "probe_id": name,
        "baseline": str(CHAMPION),
        "fixed_base": exact_manifest["probe_id"],
        "path": str(path),
        "candidate_ids": [candidate["candidate_id"] for candidate in selected],
        "actions": actions,
        "predictions": predictions,
        "prediction_delta_vs_exact": predictions - base_predictions,
        "sha256": sha256(path),
        "score_possibilities": possibilities,
    }
    (OUT / "manifests" / f"{name}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "submissions").mkdir(exist_ok=True)
    (OUT / "manifests").mkdir(exist_ok=True)

    order_ids, roots = load_submission(CHAMPION)
    base_nodes = champion_nodes(roots)
    with gzip.open(RECORDS, "rt", encoding="utf-8") as handle:
        records = json.load(handle)["test"]
    records_by_order = {row["order_id"]: row for row in records}
    exact = read_json(V37_MANIFEST)
    exact_orders = {action["order_id"] for action in exact["actions"]}

    scored = collect_scored_submissions()
    keys, matrix, rhs, _ = build_system(scored, base_nodes)
    raw_candidates = [
        candidate for candidate in collect_candidates()
        if not ({action["order_id"] for action in candidate["actions"]} & exact_orders)
    ]
    annotated = [
        annotate_candidate(candidate, keys, matrix, rhs, len(base_nodes))
        for candidate in raw_candidates
    ]
    annotated.sort(key=lambda row: (
        -int(row["delta_tp_min"] > 0),
        -row["history_rank_gain"],
        -row["maximum_information_bits"],
        -row["priority"],
    ))

    def pick(kind, count, fresh_only=False, prefer_v36=False):
        rows = [row for row in annotated if row["kind"] == kind]
        if fresh_only:
            rows = [row for row in rows if row["history_overlap"] == 0]
        rows.sort(key=lambda row: (
            -(row["source"] == "v36") if prefer_v36 else 0,
            row["source_rank"], -row["priority"],
        ))
        selected, orders = [], set(exact_orders)
        for row in rows:
            local_orders = {action["order_id"] for action in row["actions"]}
            if local_orders & orders:
                continue
            selected.append(row)
            orders |= local_orders
            if len(selected) == count:
                break
        return selected

    probes = {}
    delete_one = pick("delete", 1, fresh_only=False, prefer_v36=True)
    probes["v41_exact_plus_count_delete1"] = emit_probe(
        "v41_exact_plus_count_delete1", delete_one, exact, order_ids, roots,
        records_by_order, len(base_nodes),
    )
    probes["v41_clean_add_top8_hold"] = emit_probe(
        "v41_clean_add_top8_hold", pick("add", 8, fresh_only=True), exact,
        order_ids, roots, records_by_order, len(base_nodes),
    )
    probes["v41_clean_delete_top8_hold"] = emit_probe(
        "v41_clean_delete_top8_hold", pick("delete", 8, fresh_only=True, prefer_v36=True),
        exact, order_ids, roots, records_by_order, len(base_nodes),
    )

    swap_rows = [row for row in annotated if row["kind"] == "swap"]
    swap_rows.sort(key=lambda row: (-row["history_rank_gain"], len(row["feasible_delta_tp"])))
    if swap_rows:
        probes["v41_exact_plus_best_equation_swap_hold"] = emit_probe(
            "v41_exact_plus_best_equation_swap_hold", [swap_rows[0]], exact,
            order_ids, roots, records_by_order, len(base_nodes),
        )

    target_tp_at_exact_p = math.ceil(TARGET_SCORE * (TRUE_ROOTS + len(base_nodes)) / 2)
    report = {
        "version": "v41-equation-aware-1",
        "champion": {
            "path": str(CHAMPION), "tp": CHAMPION_TP,
            "predictions": len(base_nodes), "sha256": sha256(CHAMPION),
        },
        "exact_base": {
            "path": exact["path"], "tp": BASE_TP_AFTER_EXACT,
            "predictions": exact["predictions"], "score": exact["expected_score"],
            "sha256": exact["sha256"],
        },
        "historical_system": {
            "equations": len(matrix), "variables": len(keys),
            "rank": int(np.linalg.matrix_rank(matrix)),
        },
        "target_math": {
            "target_score": TARGET_SCORE,
            "tp_required_at_p1035": target_tp_at_exact_p,
            "tp_gap_after_exact": target_tp_at_exact_p - BASE_TP_AFTER_EXACT,
        },
        "candidate_count": len(annotated),
        "candidates": annotated,
        "probes": probes,
        "recommendation": {
            "submit_first": exact["path"],
            "submit_second_after_exact_is_verified": probes["v41_exact_plus_count_delete1"]["path"],
            "hold": [manifest["path"] for name, manifest in probes.items() if name.endswith("_hold")],
            "reason": "Only the first file has a proof-positive TP gain. The second isolates the strongest count-model deletion candidate. Larger batches remain calibration probes, not champion candidates.",
        },
    }
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "historical_system": report["historical_system"],
        "target_math": report["target_math"],
        "candidate_count": report["candidate_count"],
        "probes": {name: {
            "path": row["path"], "predictions": row["predictions"],
            "sha256": row["sha256"], "candidate_ids": row["candidate_ids"],
        } for name, row in probes.items()},
        "recommendation": report["recommendation"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
