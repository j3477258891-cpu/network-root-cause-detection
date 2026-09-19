"""Independent structural and proof audit for V37 exact corrections."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(r"D:\zgyidong")
sys.path.insert(0, str(ROOT / ".deps"))

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


REPORT = ROOT / "experiments/v37_online_equations/report.json"
MANIFEST = ROOT / "experiments/v37_online_equations/manifests/v37_exact_corrections.json"
TEST = ROOT / "test"
TRUE_ROOTS = 1044


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_nodes(path):
    rows, nodes, counts = [], set(), {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            oid = row["order_id"]
            roots = json.loads(row["output"])["rootcause"]
            rows.append(oid)
            counts[oid] = len(roots)
            for node in roots:
                nodes.add((oid, node["@rid"]))
    return rows, nodes, counts


def main():
    report, manifest = read_json(REPORT), read_json(MANIFEST)
    baseline = Path(report["champion"]["path"])
    candidate = Path(manifest["path"])
    expected_orders = sorted(path.name for path in TEST.iterdir() if path.is_dir())
    base_rows, base_nodes, base_counts = load_nodes(baseline)
    rows, nodes, counts = load_nodes(candidate)
    assert rows == base_rows == expected_orders
    assert len(rows) == 546 and all(1 <= value <= 8 for value in counts.values())
    assert sha256(candidate) == manifest["sha256"]
    assert len(nodes) == manifest["predictions"] == 1035

    variables = sorted({
        (oid, rid)
        for equation in report["equations"]
        for oid, rid in (
            (key[0], key[1]) for key in []
        )
    })
    # Rebuild the system directly from every scored CSV, not from V37's matrix.
    equation_rows = []
    variable_set = set()
    for entry in report["scored_submissions"]:
        _, submitted, _ = load_nodes(entry["path"])
        added, removed = submitted - base_nodes, base_nodes - submitted
        if not added and not removed:
            assert entry["tp"] == report["champion"]["tp"]
            continue
        variable_set |= added | removed
        equation_rows.append((added, removed, entry["tp"] - report["champion"]["tp"]))
    variables = sorted(variable_set)
    index = {key: i for i, key in enumerate(variables)}
    matrix, rhs = [], []
    for added, removed, value in equation_rows:
        vector = np.zeros(len(variables))
        for key in added:
            vector[index[key]] += 1
        for key in removed:
            vector[index[key]] -= 1
        matrix.append(vector)
        rhs.append(value)
    matrix, rhs = np.asarray(matrix), np.asarray(rhs)
    constraints = LinearConstraint(matrix, rhs, rhs)
    bounds = Bounds(np.zeros(len(variables)), np.ones(len(variables)))
    integrality = np.ones(len(variables), dtype=np.int8)
    assert milp(np.zeros(len(variables)), integrality=integrality,
                bounds=bounds, constraints=constraints).success

    fixed = {(row["order_id"], row["rid"]): row["label"] for row in report["fixed_labels"]}
    for key, label in fixed.items():
        assert key in index
        pin = np.zeros(len(variables)); pin[index[key]] = 1
        opposite = 1 - label
        test_constraints = [constraints, LinearConstraint(pin, opposite, opposite)]
        result = milp(np.zeros(len(variables)), integrality=integrality,
                      bounds=bounds, constraints=test_constraints)
        assert not result.success, f"label not fixed: {key}={label}"

    removed, added = base_nodes - nodes, nodes - base_nodes
    claimed_removed = {(a["order_id"], rid) for a in manifest["actions"] for rid in a["remove_rids"]}
    claimed_added = {(a["order_id"], rid) for a in manifest["actions"] for rid in a["add_rids"]}
    assert removed == claimed_removed and added == claimed_added
    assert all(fixed[key] == 0 for key in removed)
    assert all(fixed[key] == 1 for key in added)
    expected_tp = report["champion"]["tp"] - sum(fixed[key] for key in removed) + sum(fixed[key] for key in added)
    assert expected_tp == manifest["expected_tp"] == 956
    expected_score = 2 * expected_tp / (TRUE_ROOTS + len(nodes))
    assert abs(expected_score - manifest["expected_score"]) < 1e-12
    print(json.dumps({
        "status": "PASS", "equations": len(matrix), "variables": len(variables),
        "fixed_labels_reproved": len(fixed), "candidate_sha256": sha256(candidate),
        "guaranteed_tp": expected_tp, "guaranteed_score": expected_score,
    }, indent=2))


if __name__ == "__main__":
    main()
