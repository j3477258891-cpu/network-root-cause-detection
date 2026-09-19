"""Budget sweep over the V25 oracle ceiling.

Reuses ``oracle_audit`` (the pure theoretical-ceiling module) WITHOUT modifying
it: monkey-patches ``MAX_ACTION`` in [2..5] and records base/oracle TP, gain,
FN coverage, and the (min_gain=170, min_fn_coverage=0.80) gate verdict at each
budget.  Output: outputs/v25_oracle_budget_sweep.json
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oracle_audit as oa  # noqa: E402  (module under oracle/)


def main():
    baseline = oa.load_baseline()
    rows = []
    for budget in (2, 3, 4, 5):
        oa.MAX_ACTION = budget
        result = oa.oracle_audit(baseline)
        rows.append(
            {
                "budget": budget,
                "base_tp": result["base_tp"],
                "oracle_tp": result["oracle_tp"],
                "gain": result["gain"],
                "fn_total": result["fn_total"],
                "fn_covered": result["fn_covered"],
                "fn_coverage": result["fn_coverage"],
                "gate_passed": result["gate_passed"],
            }
        )

    output = Path(__file__).resolve().parent.parent / "outputs" / "v25_oracle_budget_sweep.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print("BUDGET SWEEP (perfect-ranker theoretical ceiling on V11 train base)")
    print("=" * 70)
    print(f"{'budget':>6} {'base':>6} {'oracle':>7} {'gain':>6} {'FN':>5} {'covered':>8} {'coverage':>9}  gate")
    for r in rows:
        print(
            f"{r['budget']:>6} {r['base_tp']:>6} {r['oracle_tp']:>7} {r['gain']:+6d} "
            f"{r['fn_total']:>5} {r['fn_covered']:>8} {r['fn_coverage']:>9.1%}  "
            f"{'PASS' if r['gate_passed'] else 'FAIL'}"
        )
    print(f"\nSaved: {output}")
    return rows


if __name__ == "__main__":
    main()
