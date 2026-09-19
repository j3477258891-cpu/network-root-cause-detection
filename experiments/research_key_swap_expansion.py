"""Research-only expansion of descriptor-key additions into same-order swaps.

This script deliberately does not create a submission CSV.  It takes the
alarm-key additions found by ``research_alarm_key_candidates.py`` and, for
orders already at the eight-root schema cap, pairs each addition with every
baseline root.  Each pair is then bounded against the real scored equation
system produced by ``research_equation_audit``.

The purpose is to test a previously unexplored route: replacing a weak
baseline root with a high-transfer-probability candidate while keeping P
constant.  Model probabilities are only priors; the equation bounds are the
authoritative online evidence.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/v60_combined_checkpoint/highest_verified_combined.csv"
ADD_CATALOG = ROOT / "experiments/research_alarm_key_candidates.json"
EQ_PATH = ROOT / "experiments/research_equation_audit/equation_system.json"
OUT = ROOT / "experiments/research_key_swap_expansion"

import sys

sys.path.insert(0, str(ROOT / "experiments/research_equation_audit"))
import audit_real_scores as equation_audit  # noqa: E402


def load_submission(path: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            payload = json.loads(row["output"])
            result[row["order_id"]] = [item["@rid"] for item in payload["rootcause"]]
    return result


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def solve(coeff: dict[int, float], matrix: csr_matrix, rhs: np.ndarray) -> tuple[int | None, int | None, str]:
    n = matrix.shape[1]
    objective = np.zeros(n, dtype=float)
    for index, value in coeff.items():
        objective[index] = value
    kwargs = {
        "integrality": np.ones(n),
        "bounds": Bounds(np.zeros(n), np.ones(n)),
        "constraints": LinearConstraint(matrix, rhs, rhs),
        "options": {"time_limit": 30.0},
    }
    lo = milp(objective, **kwargs)
    hi = milp(-objective, **kwargs)
    if not lo.success or not hi.success or lo.fun is None or hi.fun is None:
        return None, None, f"lo={lo.status}:{lo.message}; hi={hi.status}:{hi.message}"
    return int(round(float(lo.fun))), int(round(float(-hi.fun))), "optimal"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    baseline = load_submission(BASE)
    catalog = json.loads(ADD_CATALOG.read_text(encoding="utf-8"))
    additions = catalog.get("additions", [])
    records, _excluded = equation_audit.collect_records()
    alarms, _orders = equation_audit.load_test_universe()
    universe_keys = sorted(alarms)
    indices = {key: i for i, key in enumerate(universe_keys)}
    matrix, rhs, _metadata = equation_audit.build_equations(records, indices)
    equations = json.loads(EQ_PATH.read_text(encoding="utf-8"))

    # Deletion priors are used only for ranking; they do not affect bounds.
    deletion_prior: dict[tuple[str, str], float] = {}
    for item in catalog.get("deletions", []):
        deletion_prior[(item["order_id"], item["rid"])] = float(item.get("p", item.get("p_root", 0.5)))

    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str | None]] = set()
    for add in additions:
        order_id = add["order_id"]
        add_rid = add["rid"]
        if order_id not in baseline or add_rid in baseline[order_id]:
            continue
        roots = baseline[order_id]
        # A pure addition is allowed only if it stays within the 1--8 schema.
        remove_options: list[str | None] = [None] if len(roots) < 8 else list(roots)
        for remove_rid in remove_options:
            key = (order_id, add_rid, remove_rid)
            if key in seen:
                continue
            seen.add(key)
            if (order_id, add_rid) not in indices:
                continue
            coeff: dict[int, float] = {indices[(order_id, add_rid)]: 1.0}
            if remove_rid is not None:
                if (order_id, remove_rid) not in indices:
                    continue
                coeff[indices[(order_id, remove_rid)]] = -1.0
            lo, hi, status = solve(coeff, matrix, rhs)
            p_add = float(add.get("p", add.get("p_root", 0.5)))
            p_remove = 0.0 if remove_rid is None else deletion_prior.get((order_id, remove_rid), 0.5)
            candidates.append(
                {
                    "order_id": order_id,
                    "remove_rid": remove_rid,
                    "add_rid": add_rid,
                    "kind": "add" if remove_rid is None else "swap",
                    "p_add_prior": p_add,
                    "p_remove_prior": p_remove,
                    "expected_delta_prior": p_add - p_remove,
                    "add_support": add.get("support"),
                    "add_kind": add.get("kind"),
                    "equation_min_delta": lo,
                    "equation_max_delta": hi,
                    "equation_status": status,
                }
            )

    candidates.sort(
        key=lambda x: (
            x["equation_min_delta"] == 1,
            x["expected_delta_prior"],
            x["p_add_prior"],
        ),
        reverse=True,
    )
    safe = [x for x in candidates if x["equation_min_delta"] == 1]
    fixed_negative = [x for x in candidates if x["equation_min_delta"] == -1 and x["equation_max_delta"] == -1]
    # A conservative independent-order upper estimate, intentionally labelled
    # as a prior diagnostic rather than a posterior probability.
    per_order: dict[str, dict[str, Any]] = {}
    for item in candidates:
        if item["expected_delta_prior"] <= 0:
            continue
        cur = per_order.get(item["order_id"])
        if cur is None or item["expected_delta_prior"] > cur["expected_delta_prior"]:
            per_order[item["order_id"]] = item
    top_independent = sorted(per_order.values(), key=lambda x: x["expected_delta_prior"], reverse=True)
    report = {
        "version": 1,
        "baseline": {"path": str(BASE), "predictions": sum(map(len, baseline.values())), "sha256": sha256(BASE)},
        "equation_shape": equations["shape"],
        "input_additions": len(additions),
        "candidate_count": len(candidates),
        "safe_min_delta_positive_count": len(safe),
        "fixed_negative_count": len(fixed_negative),
        "candidate_kind_counts": {
            kind: sum(x["kind"] == kind for x in candidates) for kind in ("add", "swap")
        },
        "top_independent_prior_sum": sum(x["expected_delta_prior"] for x in top_independent),
        "top_independent_count": len(top_independent),
        "top_independent": top_independent[:100],
        "safe_candidates": safe,
        "fixed_negative": fixed_negative,
        "candidates": candidates,
        "note": "Research only. Priors are not online evidence; no submission file was generated.",
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("input_additions", "candidate_count", "safe_min_delta_positive_count", "fixed_negative_count", "candidate_kind_counts", "top_independent_prior_sum", "top_independent_count")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
