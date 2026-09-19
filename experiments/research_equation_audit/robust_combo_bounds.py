"""Robust joint-combination bound under the real leaderboard equations.

This is a read-only research diagnostic.  It takes the 0/1 label assignments
consistent with the public scored submissions and asks whether any *single*
set of independent V120/V121/V122 actions has a strictly positive worst-case
TP change.  The problem is a small robust integer program solved by a
constraint-generation loop:

    max_x min_{z: M z = b}  sum_j x_j q_j^T z

where ``x`` selects actions and ``z`` is a hidden 0/1 root label.  Every
adversarial label assignment found by the inner MILP contributes one master
cut.  No candidate CSV is created or modified.

The default independence policy allows at most one action per order and no
shared alarm RID.  ``--node-only`` can be used for a less restrictive
diagnostic (still no overlapping alarm nodes).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_real_scores as audit  # noqa: E402


CATALOGS = (
    (ROOT / "experiments/v120_swap_campaign/candidate_catalog.json", "v120"),
    (ROOT / "experiments/v121_equation_safe_campaign/candidate_catalog.json", "v121"),
    (ROOT / "experiments/v122_multimodel_addition_campaign/candidate_catalog.json", "v122"),
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_candidates(universe: dict[tuple[str, str], int], baseline: dict[str, list[str]]) -> list[dict[str, Any]]:
    """Load, deduplicate, and structurally validate the three catalogs."""
    merged: dict[tuple[str, str, str | None], dict[str, Any]] = {}
    for path, source in CATALOGS:
        for raw in audit.load_catalog(path, source):
            order_id = raw.get("order_id")
            add_rid = raw.get("add_rid")
            remove_rid = raw.get("remove_rid")
            kind = raw.get("kind")
            if not order_id or kind not in {"addition", "deletion", "swap"}:
                continue
            if add_rid and (order_id, add_rid) not in universe:
                continue
            if remove_rid and (order_id, remove_rid) not in universe:
                continue
            current = set(baseline.get(order_id, []))
            # Only actions that can be applied to the current 1035-root
            # baseline are admissible.  This also removes stale catalog rows.
            if kind == "addition" and (not add_rid or add_rid in current):
                continue
            if kind == "deletion" and (not remove_rid or remove_rid not in current):
                continue
            if kind == "swap" and (
                not remove_rid or not add_rid or remove_rid not in current or add_rid in current
            ):
                continue
            key = (order_id, remove_rid or "", add_rid or "")
            if key not in merged:
                item = dict(raw)
                item["catalog_sources"] = [source]
                item["catalog_ids"] = [str(raw.get("candidate_id", key))]
                merged[key] = item
            else:
                item = merged[key]
                item.setdefault("catalog_sources", []).append(source)
                item.setdefault("catalog_ids", []).append(str(raw.get("candidate_id", key)))
                # Retain the strongest available prior for descriptive
                # ranking; equations, not this prior, determine the bound.
                for field in ("nominal_p_net_gain", "conservative_p_net_gain", "model_score"):
                    try:
                        if float(raw.get(field, -1e9)) > float(item.get(field, -1e9)):
                            item[field] = raw[field]
                    except (TypeError, ValueError):
                        pass
    result = list(merged.values())
    result.sort(key=lambda x: (-float(x.get("nominal_p_net_gain") or 0.0), x["order_id"], x.get("remove_rid", ""), x.get("add_rid", "")))
    return result


def coefficient_vectors(candidates: list[dict[str, Any]], universe: dict[tuple[str, str], int], n_labels: int) -> list[dict[int, float]]:
    vectors: list[dict[int, float]] = []
    for candidate in candidates:
        coeff: dict[int, float] = {}
        order_id = candidate["order_id"]
        if candidate.get("add_rid"):
            coeff[universe[(order_id, candidate["add_rid"])]] = 1.0
        if candidate.get("remove_rid"):
            index = universe[(order_id, candidate["remove_rid"])]
            coeff[index] = coeff.get(index, 0.0) - 1.0
        vectors.append(coeff)
    return vectors


def scenario_value(coefficients: list[dict[int, float]], selected: list[int], labels: np.ndarray) -> np.ndarray:
    values = np.zeros(len(coefficients), dtype=np.float64)
    for j in selected:
        values[j] = sum(value * labels[index] for index, value in coefficients[j].items())
    return values


def solve_adversary(
    selected: list[int],
    coefficients: list[dict[int, float]],
    matrix,
    rhs,
    label_time_limit: float,
) -> tuple[int | None, np.ndarray | None, str]:
    n_labels = matrix.shape[1]
    objective = np.zeros(n_labels, dtype=np.float64)
    for j in selected:
        for index, value in coefficients[j].items():
            objective[index] += value
    result = milp(
        objective,
        integrality=np.ones(n_labels),
        bounds=Bounds(np.zeros(n_labels), np.ones(n_labels)),
        constraints=LinearConstraint(matrix, rhs, rhs),
        options={"time_limit": float(label_time_limit), "mip_rel_gap": 0},
    )
    if not result.success or result.fun is None or result.x is None:
        return None, None, f"status={result.status}; {result.message}"
    labels = np.rint(np.asarray(result.x, dtype=np.float64)).astype(np.int8)
    return int(round(float(result.fun))), labels, str(result.message)


def build_selection_groups(candidates: list[dict[str, Any]], node_only: bool) -> list[list[int]]:
    by_node: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
    by_order: defaultdict[str, list[int]] = defaultdict(list)
    for j, candidate in enumerate(candidates):
        order_id = candidate["order_id"]
        for field in ("add_rid", "remove_rid"):
            rid = candidate.get(field)
            if rid:
                by_node[(order_id, rid)].append(j)
        by_order[order_id].append(j)
    groups: list[list[int]] = [values for values in by_node.values() if len(values) > 1]
    if not node_only:
        groups.extend(values for values in by_order.values() if len(values) > 1)
    return groups


def solve_master(
    n_candidates: int,
    groups: list[list[int]],
    cuts: list[np.ndarray],
    master_time_limit: float,
    prediction_coefficients: np.ndarray | None = None,
    prediction_delta_target: int | None = None,
) -> tuple[float | None, np.ndarray | None, str]:
    """Solve max t subject to t <= c_s*x for all discovered scenarios."""
    n_vars = n_candidates + 1  # final variable is integer t
    rows = len(groups) + len(cuts) + (1 if prediction_coefficients is not None else 0)
    matrix = lil_matrix((rows, n_vars), dtype=np.float64)
    lower = np.full(rows, -np.inf, dtype=np.float64)
    upper = np.ones(rows, dtype=np.float64)
    row = 0
    for values in groups:
        for j in values:
            matrix[row, j] = 1.0
        row += 1
    for scenario in cuts:
        # t - c*x <= 0
        matrix[row, :n_candidates] = -scenario
        matrix[row, n_candidates] = 1.0
        upper[row] = 0.0
        row += 1
    if prediction_coefficients is not None:
        matrix[row, :n_candidates] = prediction_coefficients
        lower[row] = float(prediction_delta_target if prediction_delta_target is not None else 0)
        upper[row] = float(prediction_delta_target if prediction_delta_target is not None else 0)
        row += 1
    objective = np.zeros(n_vars, dtype=np.float64)
    objective[n_candidates] = -1.0
    bound = max(1, n_candidates)
    result = milp(
        objective,
        integrality=np.ones(n_vars),
        bounds=Bounds(np.r_[np.zeros(n_candidates), -float(bound)], np.r_[np.ones(n_candidates), float(bound)]),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options={"time_limit": float(master_time_limit), "mip_rel_gap": 0},
    )
    if not result.success or result.x is None:
        return None, None, f"status={result.status}; {result.message}"
    return float(round(float(result.x[n_candidates]))), np.rint(result.x[:n_candidates]).astype(np.int8), str(result.message)


def robust_search(
    candidates: list[dict[str, Any]],
    coefficients: list[dict[int, float]],
    equations,
    rhs,
    groups: list[list[int]],
    master_time_limit: float,
    label_time_limit: float,
    max_iterations: int,
    prediction_coefficients: np.ndarray | None = None,
    prediction_delta_target: int | None = None,
) -> dict[str, Any]:
    cuts: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for iteration in range(1, max_iterations + 1):
        t_value, x, master_message = solve_master(
            len(candidates), groups, cuts, master_time_limit,
            prediction_coefficients, prediction_delta_target,
        )
        if t_value is None or x is None:
            return {
                "status": "master_infeasible_or_timeout",
                "iterations": iteration,
                "history": history,
                "message": master_message,
            }
        selected = [j for j, value in enumerate(x) if value > 0]
        minimum, labels, adversary_message = solve_adversary(
            selected, coefficients, equations, rhs, label_time_limit
        )
        if minimum is None or labels is None:
            return {
                "status": "adversary_infeasible_or_timeout",
                "iterations": iteration,
                "history": history,
                "master_t": t_value,
                "selected_count": len(selected),
                "message": adversary_message,
            }
        scenario = scenario_value(coefficients, list(range(len(candidates))), labels)
        key = tuple(int(v) for v in labels)
        history.append(
            {
                "iteration": iteration,
                "master_t": int(t_value),
                "adversary_min": int(minimum),
                "selected_count": len(selected),
                "selected_ids": [candidates[j].get("candidate_id") for j in selected],
                "scenario_sum_check": int(round(float(np.dot(scenario, x)))),
                "master_message": master_message,
                "adversary_message": adversary_message,
            }
        )
        if best is None or minimum > int(best["verified_worst_case_delta"]):
            best = {
                "verified_worst_case_delta": int(minimum),
                "master_t": int(t_value),
                "selected": selected,
                "selected_ids": [candidates[j].get("candidate_id") for j in selected],
            }
        # If the exact adversary cannot beat t, this master solution is
        # globally robust (the inner MILP was solved over all labels).
        if minimum >= t_value:
            best = {
                "verified_worst_case_delta": int(minimum),
                "master_t": int(t_value),
                "selected": selected,
                "selected_ids": [candidates[j].get("candidate_id") for j in selected],
            }
            return {
                "status": "converged",
                "iterations": iteration,
                "history": history,
                "best": best,
                "cut_count": len(cuts),
            }
        if key in seen:
            return {
                "status": "repeated_adversarial_scenario",
                "iterations": iteration,
                "history": history,
                "best": best,
                "cut_count": len(cuts),
            }
        seen.add(key)
        cuts.append(scenario)
    return {
        "status": "iteration_limit",
        "iterations": max_iterations,
        "history": history,
        "best": best,
        "cut_count": len(cuts),
    }


def enrich_result(result: dict[str, Any], candidates: list[dict[str, Any]], robust_target: int = 27) -> dict[str, Any]:
    selected = result.get("best", {}).get("selected", []) if isinstance(result.get("best"), dict) else []
    rows = []
    for j in selected:
        c = candidates[int(j)]
        rows.append(
            {
                "candidate_id": c.get("candidate_id"),
                "catalog_ids": c.get("catalog_ids"),
                "catalog_sources": c.get("catalog_sources"),
                "kind": c.get("kind"),
                "order_id": c.get("order_id"),
                "remove_rid": c.get("remove_rid"),
                "add_rid": c.get("add_rid"),
                "nominal_p_net_gain": c.get("nominal_p_net_gain"),
                "conservative_p_net_gain": c.get("conservative_p_net_gain"),
                "v119_prior": c.get("v119_prior"),
                "v11_mean": c.get("v11_mean"),
                "consensus_score": c.get("consensus_score"),
                "equation_min_delta": c.get("equation_min_delta"),
                "equation_max_delta": c.get("equation_max_delta"),
            }
        )
    result["selected_candidates"] = rows
    delta_p = sum(1 if c.get("kind") == "addition" else -1 if c.get("kind") == "deletion" else 0 for c in rows)
    worst_delta = result.get("best", {}).get("verified_worst_case_delta") if isinstance(result.get("best"), dict) else None
    if worst_delta is not None:
        denominator = 1044 + 1035 + delta_p
        worst_f1 = 2.0 * (956 + int(worst_delta)) / denominator
        required_tp = int(np.ceil(0.945 * denominator / 2.0))
        result["selected_prediction_delta"] = int(delta_p)
        result["worst_case_f1_if_selected"] = worst_f1
        result["required_tp_for_0945_at_selected_p"] = required_tp
        result["required_delta_tp_for_0945_at_selected_p"] = required_tp - 956
    result["target_guaranteed_delta"] = robust_target
    result["can_guarantee_target"] = bool(
        result.get("best", {}).get("verified_worst_case_delta", -10**9) >= robust_target
    ) if isinstance(result.get("best"), dict) else False
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-time-limit", type=float, default=20.0)
    parser.add_argument("--label-time-limit", type=float, default=15.0)
    parser.add_argument("--max-iterations", type=int, default=80)
    parser.add_argument("--node-only", action="store_true", help="Do not enforce at-most-one action per order")
    parser.add_argument(
        "--kind",
        choices=("all", "swap", "addition", "deletion"),
        default="all",
        help="Restrict the robust search to one action kind (default: all)",
    )
    parser.add_argument("--output", type=Path, default=HERE / "robust_combo_bounds.json")
    args = parser.parse_args()

    records, _ = audit.collect_records()
    alarms, _ = audit.load_test_universe()
    keys = sorted(alarms)
    universe = {key: index for index, key in enumerate(keys)}
    equations, rhs, equation_meta = audit.build_equations(records, universe)
    baseline = audit.load_submission(audit.BASE_PATH)
    candidates = load_candidates(universe, baseline)
    if args.kind != "all":
        candidates = [candidate for candidate in candidates if candidate.get("kind") == args.kind]
    coefficients = coefficient_vectors(candidates, universe, equations.shape[1])
    groups = build_selection_groups(candidates, args.node_only)
    result = robust_search(
        candidates,
        coefficients,
        equations,
        rhs,
        groups,
        args.master_time_limit,
        args.label_time_limit,
        args.max_iterations,
    )
    result = enrich_result(result, candidates)
    payload = {
        "version": 1,
        "source_policy": "37 public scored equations; V120/V121/V122 catalogs only",
        "equation_shape": [int(equations.shape[0]), int(equations.shape[1])],
        "equation_feasible": True,
        "candidate_count": len(candidates),
        "candidate_kind_counts": {
            kind: sum(c.get("kind") == kind for c in candidates)
            for kind in ("swap", "addition", "deletion")
        },
        "independence_policy": "node_only" if args.node_only else "node_and_order",
        "disjoint_constraint_count": len(groups),
        "target": {"f1": 0.945, "baseline_tp": 956, "predictions": 1035, "required_delta_tp": 27},
        "result": result,
        "equation_sources": equation_meta,
        "interpretation": [
            "A converged verified_worst_case_delta is the minimum over all binary labels satisfying every scored equation.",
            "A value <=0 means the public equations do not certify a positive TP gain for any selected independent action set.",
            "This is a robust evidence bound, not a posterior probability and not a submission recommendation by itself.",
            "No submission CSV is generated by this diagnostic.",
        ],
    }
    write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
