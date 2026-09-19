"""Equation bounds for unique V30 manifest add/delete actions (read-only)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_real_scores as audit  # noqa: E402
import robust_broad_catalogs as broad  # noqa: E402


def main():
    records, _ = audit.collect_records()
    alarms, _ = audit.load_test_universe()
    universe = {key: i for i, key in enumerate(sorted(alarms))}
    equations, rhs, _ = audit.build_equations(records, universe)
    baseline = audit.load_submission(audit.BASE_PATH)
    candidates = broad.load_v30_manifests(ROOT / "experiments/v30_meta_stack", universe, baseline)
    rows = []
    for i, c in enumerate(candidates):
        coeff = {}
        if c.get("add_rid"):
            coeff[universe[(c["order_id"], c["add_rid"])]] = 1
        if c.get("remove_rid"):
            coeff[universe[(c["order_id"], c["remove_rid"])]] = coeff.get(universe[(c["order_id"], c["remove_rid"])], 0) - 1
        bounds = audit.solve_min_max(coeff, equations, rhs, 1.0)
        rows.append({**c, "equation_bounds": bounds})
    rows.sort(key=lambda x: (x.get("kind") != "addition", -(float(x.get("nominal_p_net_gain") or 0.0))))
    output = {
        "version": 1, "candidate_count": len(rows),
        "fixed_positive_count": sum(r["equation_bounds"].get("min") == 1 for r in rows),
        "fixed_negative_count": sum(r["equation_bounds"].get("min") == -1 and r["equation_bounds"].get("max") == -1 for r in rows),
        "rows": rows,
        "interpretation": ["V30 expected_gain is a model margin, not a calibrated probability.", "No submission CSV generated."],
    }
    path = HERE / "v30_action_bounds.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: output[k] for k in ("candidate_count", "fixed_positive_count", "fixed_negative_count")}, ensure_ascii=False, indent=2))
    print(json.dumps(rows[:20], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
