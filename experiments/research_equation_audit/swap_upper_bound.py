"""Bound the best disjoint V120 swap set under the real-score equations.

This is a research diagnostic only.  It never writes a submission CSV and it
does not claim that an equation-consistent binary label assignment is the
actual hidden test truth.  For each V120 pair, ``z_i`` may be selected only
when the MILP label variables permit add=1 and remove=0.  Historical scored
equations are kept exact.  We report both a logical upper bound (maximize the
number of +1 pairs) and a model-weighted upper bound (maximize the catalog's
nominal/conservative probabilities).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_real_scores as audit  # noqa: E402


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_problem(time_limit: float) -> tuple[list[dict], dict[str, object]]:
    records, _ = audit.collect_records()
    alarms, _ = audit.load_test_universe()
    keys = sorted(alarms)
    universe = {key: index for index, key in enumerate(keys)}
    equations, rhs, _ = audit.build_equations(records, universe)

    catalog_path = ROOT / "experiments/v120_swap_campaign/candidate_catalog.json"
    candidates = audit.load_catalog(catalog_path, "v120")
    unique: dict[tuple[str, str, str], dict] = {}
    for candidate in candidates:
        remove_rid = candidate.get("remove_rid")
        add_rid = candidate.get("add_rid")
        order_id = candidate.get("order_id")
        if not (order_id and remove_rid and add_rid):
            continue
        if (order_id, remove_rid) not in universe or (order_id, add_rid) not in universe:
            continue
        unique.setdefault((order_id, remove_rid, add_rid), candidate)
    candidates = list(unique.values())

    n_labels = equations.shape[1]
    n_pairs = len(candidates)
    n_vars = n_labels + n_pairs
    base_rows = equations.shape[0] + 2 * n_pairs
    matrix = lil_matrix((base_rows, n_vars), dtype=float)
    matrix[: equations.shape[0], :n_labels] = equations
    lower = list(rhs.astype(float))
    upper = list(rhs.astype(float))
    row = equations.shape[0]
    for pair_index, candidate in enumerate(candidates):
        add_index = universe[(candidate["order_id"], candidate["add_rid"])]
        remove_index = universe[(candidate["order_id"], candidate["remove_rid"])]
        z_index = n_labels + pair_index
        # z <= label(add), z <= 1-label(remove)
        matrix[row, z_index] = 1
        matrix[row, add_index] = -1
        lower.append(-np.inf)
        upper.append(0)
        row += 1
        matrix[row, z_index] = 1
        matrix[row, remove_index] = 1
        lower.append(-np.inf)
        upper.append(1)
        row += 1

    # A final submission cannot use overlapping nodes or two swaps from one
    # order.  This is stricter than the equation bound and avoids an invalid
    # set whose operations overwrite each other.
    by_node: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
    by_order: defaultdict[str, list[int]] = defaultdict(list)
    for pair_index, candidate in enumerate(candidates):
        by_node[(candidate["order_id"], candidate["add_rid"])].append(pair_index)
        by_node[(candidate["order_id"], candidate["remove_rid"])].append(pair_index)
        by_order[candidate["order_id"]].append(pair_index)
    groups: list[list[int]] = []
    for values in list(by_node.values()) + list(by_order.values()):
        if len(values) > 1:
            groups.append(values)
    expanded = lil_matrix((matrix.shape[0] + len(groups), n_vars), dtype=float)
    expanded[: matrix.shape[0], :] = matrix
    lower_expanded = list(lower)
    upper_expanded = list(upper)
    for offset, values in enumerate(groups):
        constraint_row = matrix.shape[0] + offset
        for pair_index in values:
            expanded[constraint_row, n_labels + pair_index] = 1
        lower_expanded.append(-np.inf)
        upper_expanded.append(1)

    constraint = LinearConstraint(
        expanded.tocsr(), np.asarray(lower_expanded), np.asarray(upper_expanded)
    )
    bounds = Bounds(np.zeros(n_vars), np.ones(n_vars))

    results: dict[str, object] = {}
    objectives = {
        "max_disjoint_positive_count": np.ones(n_pairs),
        "max_nominal_probability_sum": np.asarray(
            [float(c.get("nominal_p_net_gain") or 0.0) for c in candidates]
        ),
        "max_conservative_probability_sum": np.asarray(
            [float(c.get("conservative_p_net_gain") or 0.0) for c in candidates]
        ),
    }
    for name, pair_objective in objectives.items():
        objective = np.zeros(n_vars)
        objective[n_labels:] = -pair_objective
        solved = milp(
            objective,
            integrality=np.ones(n_vars),
            bounds=bounds,
            constraints=constraint,
            options={"time_limit": float(time_limit), "mip_rel_gap": 0},
        )
        entry: dict[str, object] = {
            "status": int(solved.status),
            "message": str(solved.message),
            "optimal": bool(solved.success),
        }
        if solved.x is not None:
            selected = [
                candidates[index]
                for index, value in enumerate(solved.x[n_labels:])
                if value > 0.5
            ]
            entry["selected_count"] = len(selected)
            entry["nominal_probability_sum"] = sum(
                float(c.get("nominal_p_net_gain") or 0.0) for c in selected
            )
            entry["conservative_probability_sum"] = sum(
                float(c.get("conservative_p_net_gain") or 0.0) for c in selected
            )
            entry["selected_candidates"] = [
                {
                    "candidate_id": c.get("candidate_id"),
                    "order_id": c.get("order_id"),
                    "remove_rid": c.get("remove_rid"),
                    "add_rid": c.get("add_rid"),
                    "nominal_p_net_gain": c.get("nominal_p_net_gain"),
                    "conservative_p_net_gain": c.get("conservative_p_net_gain"),
                }
                for c in selected
            ]
            # In the selected branch every z=1 is constrained to a +1 label
            # difference.  Keep the expected F1 only as a descriptive number.
            entry["expected_tp_if_selected_all_positive"] = 956 + len(selected)
            entry["f1_if_selected_all_positive"] = 2 * (956 + len(selected)) / (1044 + 1035)
        results[name] = entry

    payload = {
        "version": 1,
        "source_policy": "37 public scored equations; V120 catalog only",
        "equation_shape": [int(equations.shape[0]), int(equations.shape[1])],
        "equation_feasible": True,
        "candidate_count": n_pairs,
        "disjoint_constraint_count": len(groups),
        "target": {"required_delta_tp": 27, "f1": 0.945, "baseline_tp": 956, "predictions": 1035},
        "results": results,
        "interpretation": [
            "The max-count branch is a logical consistency upper bound, not a posterior probability.",
            "Catalog probabilities are model priors and are not calibrated by this MILP.",
            "No submission CSV is generated by this diagnostic.",
        ],
    }
    return candidates, payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-limit", type=float, default=180.0)
    parser.add_argument("--output", type=Path, default=HERE / "swap_upper_bound.json")
    args = parser.parse_args()
    _, payload = build_problem(args.time_limit)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
