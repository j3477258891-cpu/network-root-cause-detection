"""Maximize worst-case F1 over candidate combinations (read-only).

For each prediction-count change ``k`` this enumerates the robust maximum
minimum TP gain under the 37 scored equations, then converts it to the actual
F1 denominator.  This prevents an addition-only TP gain from being mistaken
for an F1 gain.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import audit_real_scores as audit
import robust_combo_bounds as robust

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def run_kind(kind: str, k_values: range):
    records, _ = audit.collect_records()
    alarms, _ = audit.load_test_universe()
    universe = {key: i for i, key in enumerate(sorted(alarms))}
    equations, rhs, _ = audit.build_equations(records, universe)
    baseline = audit.load_submission(audit.BASE_PATH)
    candidates = robust.load_candidates(universe, baseline)
    if kind != "all":
        candidates = [c for c in candidates if c.get("kind") == kind]
    vectors = robust.coefficient_vectors(candidates, universe, equations.shape[1])
    groups = robust.build_selection_groups(candidates, False)
    pcoeff = np.asarray([
        1 if c.get("kind") == "addition" else -1 if c.get("kind") == "deletion" else 0
        for c in candidates
    ], dtype=np.float64)
    rows = []
    for k in k_values:
        result = robust.robust_search(
            candidates, vectors, equations, rhs, groups,
            master_time_limit=8.0, label_time_limit=8.0, max_iterations=30,
            prediction_coefficients=pcoeff, prediction_delta_target=int(k),
        )
        best = result.get("best", {}) if isinstance(result.get("best"), dict) else {}
        t = best.get("verified_worst_case_delta")
        entry = {
            "kind": kind, "prediction_delta": int(k), "status": result.get("status"),
            "iterations": result.get("iterations"), "worst_case_delta_tp": t,
            "selected_count": len(best.get("selected", [])),
        }
        if t is not None:
            denominator = 1044 + 1035 + int(k)
            entry["worst_case_f1"] = 2.0 * (956 + int(t)) / denominator
            entry["delta_f1_vs_baseline"] = entry["worst_case_f1"] - 2.0 * 956 / 2079
            entry["target_reached"] = entry["worst_case_f1"] >= 0.945
        rows.append(entry)
    return rows


def main():
    # Official catalogs contain 184 additions and 144 swaps, so k=0..25 is
    # enough to cover every plausible F1-improving addition portfolio.
    output = {
        "official_all": run_kind("all", range(0, 26)),
        "official_swap": run_kind("swap", range(0, 1)),
    }
    path = HERE / "robust_f1_bound.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    best = max((row for rows in output.values() for row in rows if "worst_case_f1" in row), key=lambda row: row["worst_case_f1"])
    print(json.dumps({"best": best, "rows": output}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
