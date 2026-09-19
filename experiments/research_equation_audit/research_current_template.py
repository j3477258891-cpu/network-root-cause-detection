"""Compare strict canonical-template predictions with the current baseline.

This research-only script checks whether V39's high-accuracy train analogue
would actually change the *current* 1,035-root test file.  It records action
differences and their 37-equation min/max TP bounds; no CSV is emitted.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(HERE))
from build_candidate_catalog import canonical_template_prediction, prepare_order  # type: ignore  # noqa: E402
import audit_real_scores as audit  # noqa: E402


def order_sites(order):
    # Match V39's site-level disjointness key without importing campaign code.
    from build_actions import site_key
    return frozenset(site_key(node) for node in order["alarms"])


def main():
    train_orders = [prepare_order(path, True) for path in sorted((ROOT / "train").iterdir()) if path.is_dir()]
    test_orders = [prepare_order(path, False) for path in sorted((ROOT / "test").iterdir()) if path.is_dir()]
    groups = defaultdict(list)
    for order in train_orders:
        groups[order["signature"]].append(order)
    baseline = audit.load_submission(ROOT / "experiments/v60_combined_checkpoint/highest_verified_combined.csv")
    records, _ = audit.collect_records()
    alarms, _ = audit.load_test_universe()
    universe = {key: i for i, key in enumerate(sorted(alarms))}
    equations, rhs, _ = audit.build_equations(records, universe)
    rows = []
    for order in test_orders:
        refs = groups.get(order["signature"], [])
        prediction = canonical_template_prediction(order, refs)
        if prediction is None:
            continue
        selected = set(prediction["selected"])
        current = set(baseline.get(order["id"], []))
        remove = sorted(current - selected)
        add = sorted(selected - current)
        if not remove and not add:
            continue
        if any((order["id"], rid) not in universe for rid in remove + add):
            continue
        coeff = {}
        for rid in add:
            coeff[universe[(order["id"], rid)]] = coeff.get(universe[(order["id"], rid)], 0) + 1
        for rid in remove:
            coeff[universe[(order["id"], rid)]] = coeff.get(universe[(order["id"], rid)], 0) - 1
        bounds = audit.solve_min_max(coeff, equations, rhs, 5.0)
        rows.append({
            "order_id": order["id"], "support": len(refs),
            "sites": len(set().union(*(order_sites(ref) for ref in refs))) if refs else 0,
            "current_count": len(current), "template_count": len(selected),
            "remove_rids": remove, "add_rids": add, "delta_p": len(add) - len(remove),
            "equation_bounds": bounds,
        })
    rows.sort(key=lambda row: (-(row["equation_bounds"].get("min") or -99), -row["support"], row["order_id"]))
    output = {
        "version": 1, "train_orders": len(train_orders), "test_orders": len(test_orders),
        "candidate_action_orders": len(rows), "candidate_action_total": sum(len(r["remove_rids"]) + len(r["add_rids"]) for r in rows),
        "equation_shape": [int(equations.shape[0]), int(equations.shape[1])],
        "min_positive_count": sum(r["equation_bounds"].get("min") == 1 for r in rows),
        "fixed_negative_count": sum(r["equation_bounds"].get("min") == -1 and r["equation_bounds"].get("max") == -1 for r in rows),
        "rows": rows,
        "interpretation": ["Canonical-template train exactness is not test truth.", "No submission CSV generated."],
    }
    path = HERE / "current_template_actions.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: output[k] for k in ("candidate_action_orders", "candidate_action_total", "min_positive_count", "fixed_negative_count")}, ensure_ascii=False, indent=2))
    print(json.dumps(rows[:10], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
