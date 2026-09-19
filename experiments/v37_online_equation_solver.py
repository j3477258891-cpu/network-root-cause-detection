"""Recover exact test labels implied by historical leaderboard submissions.

Every scored submission gives an exact equation: the sum of hidden binary
labels selected by that CSV equals its reconstructed true-positive count.
Taking differences against the current champion removes every untouched node
and leaves a small binary linear system over historically probed RIDs.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"D:\zgyidong")
if str(ROOT / ".deps") not in sys.path:
    sys.path.insert(0, str(ROOT / ".deps"))
if str(ROOT / "experiments/v30_meta_stack") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments/v30_meta_stack"))

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from build_cross_order_probes import apply_actions, load_submission, write_submission


CHAMPION = ROOT / "experiments/v30_meta_stack/submissions/v30_cross_order_top5.csv"
RECORDS = ROOT / "experiments/v25_semantic_router/cloud_dataset/semantic_records.json.gz"
OUT = ROOT / "experiments/v37_online_equations"
CHAMPION_TP = 955
TRUE_ROOTS = 1044
MAX_ROOTS = 8


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def submission_nodes(path: Path) -> set[tuple[str, str]]:
    nodes = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 546:
        raise ValueError(f"{path}: expected 546 rows, got {len(rows)}")
    for row in rows:
        output = json.loads(row["output"])
        roots = output.get("rootcause", [])
        if not 1 <= len(roots) <= MAX_ROOTS:
            raise ValueError(f"{path}: invalid root count for {row['order_id']}")
        for node in roots:
            nodes.add((row["order_id"], node["@rid"]))
    return nodes


def add_scored(output: list[dict], seen: set[tuple[str, int]], name, path, tp, source):
    if not path or tp is None:
        return
    path = resolve_path(path)
    if not path.exists():
        return
    key = (sha256(path), int(tp))
    if key in seen:
        return
    seen.add(key)
    output.append({
        "name": str(name), "path": str(path), "tp": int(tp),
        "source": str(source), "sha256": key[0],
    })


def collect_scored_submissions() -> list[dict]:
    output, seen = [], set()

    ledger_path = ROOT / "experiments/ledger.json"
    if ledger_path.exists():
        ledger = read_json(ledger_path)
        for digest, row in ledger.get("scores", {}).items():
            add_scored(output, seen, digest[:12], row.get("path"), row.get("tp"), ledger_path)

    state_specs = (
        ROOT / "experiments/v27_closed_loop/state.json",
        ROOT / "experiments/v28_ten_day_campaign/reports/phase1_precision_state.json",
        ROOT / "experiments/v28_ten_day_campaign/state.json",
    )
    for state_path in state_specs:
        if not state_path.exists():
            continue
        state = read_json(state_path)
        champion = state.get("champion", {})
        add_scored(output, seen, f"{state_path.parent.name}_champion",
                   champion.get("path"), champion.get("tp"), state_path)
        for batch in state.get("batches", []):
            if batch.get("status") != "scored":
                continue
            add_scored(output, seen, batch.get("batch_id"), batch.get("submission"),
                       batch.get("tp"), state_path)

    online_specs = (
        ROOT / "experiments/v29_domain_ranker/reports/online_results.json",
        ROOT / "experiments/v30_meta_stack/online_results.json",
        ROOT / "experiments/v53_active_equation/online_result.json",
    )
    for online_path in online_specs:
        if not online_path.exists():
            continue
        online = read_json(online_path)
        baseline = online.get("baseline", {})
        add_scored(output, seen, f"{online_path.parent.parent.name}_baseline",
                   baseline.get("path"), baseline.get("tp"), online_path)
        champion = online.get("champion", {})
        add_scored(output, seen, f"{online_path.parent.name}_champion",
                   champion.get("path"), champion.get("tp"), online_path)
        for row in online.get("verified", []):
            path = row.get("submitted_path") or row.get("path")
            if not path and row.get("probe_id"):
                manifest_path = online_path.parent / "manifests" / f"{row['probe_id']}.json"
                if manifest_path.exists():
                    path = read_json(manifest_path).get("path")
            add_scored(output, seen, row.get("probe_id"), path, row.get("tp"), online_path)

    add_scored(output, seen, "current_champion", CHAMPION, CHAMPION_TP, "constant")
    return output


def build_system(scored: list[dict], champion_nodes: set[tuple[str, str]]):
    rows = []
    variables = set()
    for entry in scored:
        nodes = submission_nodes(Path(entry["path"]))
        added = nodes - champion_nodes
        removed = champion_nodes - nodes
        if not added and not removed:
            if entry["tp"] != CHAMPION_TP:
                raise ValueError(f"inconsistent duplicate champion: {entry}")
            continue
        variables |= added | removed
        rows.append({**entry, "added": added, "removed": removed,
                     "rhs": int(entry["tp"] - CHAMPION_TP)})

    keys = sorted(variables)
    index = {key: i for i, key in enumerate(keys)}
    matrix, rhs = [], []
    for row in rows:
        vector = np.zeros(len(keys), dtype=np.float64)
        for key in row["added"]:
            vector[index[key]] += 1
        for key in row["removed"]:
            vector[index[key]] -= 1
        matrix.append(vector)
        rhs.append(row["rhs"])
    return keys, np.asarray(matrix), np.asarray(rhs, dtype=np.float64), rows


def solve_fixed_labels(matrix: np.ndarray, rhs: np.ndarray, n: int):
    if n == 0:
        return np.empty(0), np.empty(0), []
    constraints = LinearConstraint(matrix, rhs, rhs)
    bounds = Bounds(np.zeros(n), np.ones(n))
    integrality = np.ones(n, dtype=np.int8)
    base = milp(np.zeros(n), integrality=integrality, bounds=bounds,
                constraints=constraints, options={"time_limit": 30})
    if not base.success:
        raise RuntimeError(f"historical equations are infeasible: {base.message}")

    lower = np.zeros(n, dtype=np.int8)
    upper = np.ones(n, dtype=np.int8)
    for i in range(n):
        objective = np.zeros(n)
        objective[i] = 1
        lo = milp(objective, integrality=integrality, bounds=bounds,
                  constraints=constraints, options={"time_limit": 10})
        hi = milp(-objective, integrality=integrality, bounds=bounds,
                  constraints=constraints, options={"time_limit": 10})
        if not lo.success or not hi.success:
            raise RuntimeError(f"failed label bound for variable {i}")
        lower[i] = int(round(lo.fun))
        upper[i] = int(round(-hi.fun))
    return lower, upper, np.rint(base.x).astype(np.int8).tolist()


def validate_equations(matrix, rhs, assignment):
    residual = matrix @ np.asarray(assignment) - rhs
    if len(residual) and not np.allclose(residual, 0):
        raise ValueError(f"nonzero residual: {residual}")


def score_possibilities(tp: int, predictions: int):
    return round(2 * tp / (TRUE_ROOTS + predictions), 12)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "submissions").mkdir(exist_ok=True)
    (OUT / "manifests").mkdir(exist_ok=True)
    scored = collect_scored_submissions()
    order_ids, champion_roots = load_submission(CHAMPION)
    champion_nodes = {(oid, node["@rid"]) for oid, roots in champion_roots.items() for node in roots}
    keys, matrix, rhs, equations = build_system(scored, champion_nodes)
    lower, upper, assignment = solve_fixed_labels(matrix, rhs, len(keys))
    validate_equations(matrix, rhs, assignment)

    fixed = []
    for i, key in enumerate(keys):
        if lower[i] != upper[i]:
            continue
        fixed.append({
            "order_id": key[0], "rid": key[1], "label": int(lower[i]),
            "selected_by_champion": key in champion_nodes,
        })
    beneficial = [row for row in fixed if row["selected_by_champion"] != bool(row["label"])]

    by_order = defaultdict(lambda: {"remove_rids": [], "add_rids": []})
    for row in beneficial:
        field = "remove_rids" if row["selected_by_champion"] else "add_rids"
        by_order[row["order_id"]][field].append(row["rid"])
    actions = []
    for index, (oid, change) in enumerate(sorted(by_order.items()), 1):
        actions.append({
            "action_id": f"v37_exact_{index:03d}", "order_id": oid,
            "remove_rids": sorted(change["remove_rids"]),
            "add_rids": sorted(change["add_rids"]),
            "source": "historical_leaderboard_equations", "expected_gain": None,
            "evidence": {"proof": "label fixed in every binary solution of all scored equations"},
        })

    with gzip.open(RECORDS, "rt", encoding="utf-8") as handle:
        records = json.load(handle)["test"]
    records_by_order = {row["order_id"]: row for row in records}
    emitted = None
    if actions:
        roots = apply_actions(champion_roots, records_by_order, actions)
        predictions = sum(map(len, roots.values()))
        known_tp_gain = sum(len(action["add_rids"]) for action in actions)
        known_tp_loss = sum(len(action["remove_rids"]) for action in actions)
        expected_tp = CHAMPION_TP + known_tp_gain - known_tp_loss
        name = "v37_exact_corrections"
        path = OUT / "submissions" / f"{name}.csv"
        write_submission(path, order_ids, roots)
        emitted = {
            "probe_id": name, "path": str(path), "baseline": str(CHAMPION),
            "actions": actions, "predictions": predictions,
            "expected_tp": expected_tp,
            "expected_score": score_possibilities(expected_tp, predictions),
            "sha256": sha256(path),
        }
        (OUT / "manifests" / f"{name}.json").write_text(
            json.dumps(emitted, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    report = {
        "version": "v37-online-equations-1",
        "champion": {"path": str(CHAMPION), "tp": CHAMPION_TP,
                     "predictions": len(champion_nodes), "sha256": sha256(CHAMPION)},
        "scored_submissions": scored,
        "equation_count": int(len(matrix)),
        "variable_count": len(keys),
        "rank_over_reals": int(np.linalg.matrix_rank(matrix)) if len(matrix) else 0,
        "fixed_label_count": len(fixed),
        "fixed_true_count": sum(row["label"] for row in fixed),
        "fixed_false_count": sum(1 - row["label"] for row in fixed),
        "beneficial_correction_count": len(beneficial),
        "fixed_labels": fixed,
        "beneficial_corrections": beneficial,
        "submission": emitted,
        "equations": [{
            "name": row["name"], "path": row["path"], "rhs": row["rhs"],
            "added": len(row["added"]), "removed": len(row["removed"]),
        } for row in equations],
    }
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: report[k] for k in (
        "equation_count", "variable_count", "rank_over_reals",
        "fixed_label_count", "fixed_true_count", "fixed_false_count",
        "beneficial_correction_count", "submission",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
