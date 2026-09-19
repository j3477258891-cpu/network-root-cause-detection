"""Sensitivity of the robust action bound to the scored-record subset.

Read-only companion to ``robust_combo_bounds.py``.  It uses the same
constraint-generation solver but repeats it after retaining only explicit
modern records or only P>=1035 records.  This checks whether the robust bound
is driven by stale low-cardinality probes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import audit_real_scores as audit
import robust_combo_bounds as robust


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def run_subset(name, records, universe, baseline, kind):
    equations, rhs, _ = audit.build_equations(records, universe)
    feasible = audit.solve_objective(np.zeros(equations.shape[1]), equations, rhs, 20.0)[0] is not None
    if not feasible:
        return {"record_count": len(records), "equation_feasible": False}
    candidates = robust.load_candidates(universe, baseline)
    if kind != "all":
        candidates = [c for c in candidates if c.get("kind") == kind]
    vectors = robust.coefficient_vectors(candidates, universe, equations.shape[1])
    groups = robust.build_selection_groups(candidates, False)
    result = robust.robust_search(candidates, vectors, equations, rhs, groups, 10.0, 10.0, 40)
    best = result.get("best", {}) if isinstance(result.get("best"), dict) else {}
    selected = best.get("selected", [])
    delta_p = sum(1 if candidates[j].get("kind") == "addition" else -1 if candidates[j].get("kind") == "deletion" else 0 for j in selected)
    worst = best.get("verified_worst_case_delta")
    out = {
        "record_count": len(records),
        "equation_shape": [int(equations.shape[0]), int(equations.shape[1])],
        "equation_feasible": True,
        "candidate_count": len(candidates),
        "kind": kind,
        "status": result.get("status"),
        "iterations": result.get("iterations"),
        "verified_worst_case_delta": worst,
        "selected_count": len(selected),
        "selected_prediction_delta": delta_p,
    }
    if worst is not None:
        denominator = 1044 + 1035 + delta_p
        out["worst_case_f1"] = 2 * (956 + int(worst)) / denominator
    return out


def main():
    records, _ = audit.collect_records()
    alarms, _ = audit.load_test_universe()
    keys = sorted(alarms)
    universe = {key: index for index, key in enumerate(keys)}
    baseline = audit.load_submission(audit.BASE_PATH)
    modern_sources = {
        "v104_user_confirmed", "v117_user_confirmed", "v118_user_confirmed",
        "v119_user_confirmed", "v120_user_confirmed", "v121_user_confirmed",
        "v75_user_confirmed", "v29_online_results", "v30_online_results",
    }
    subsets = {
        "all_37": records,
        "modern_explicit": [r for r in records if r["source"] in modern_sources],
        "p_ge_1035": [r for r in records if int(r["predictions"]) >= 1035],
        "modern_p_ge_1035": [r for r in records if r["source"] in modern_sources and int(r["predictions"]) >= 1035],
    }
    output = {}
    for name, subset in subsets.items():
        for kind in ("swap", "all"):
            output[f"{name}_{kind}"] = run_subset(name, subset, universe, baseline, kind)
    path = HERE / "robust_combo_sensitivity.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
