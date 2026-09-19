"""Find and emit the strongest two-node, fixed-count equation probe.

Unlike a broad model batch, every V55 candidate makes one legal addition and
one legal deletion in different orders.  The prediction count therefore stays
at the verified V37 value.  Candidates already implied by the historical
leaderboard equations are discarded; the selected pair maximizes the expected
number of labels fixed by its new exact leaderboard equation.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
V30 = EXP / "v30_meta_stack"
for value in (EXP, V30):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from build_cross_order_probes import apply_actions, load_submission, write_submission
from v37_online_equation_solver import (
    CHAMPION, CHAMPION_TP, TRUE_ROOTS, build_system, collect_scored_submissions,
    solve_fixed_labels,
)
from v45_constrained_map_inference import priors
from v53_active_equation_probe import corrections_from_fixed, checkpoint_math, prior_delta_distribution


RECORDS = EXP / "v25_semantic_router/cloud_dataset/semantic_records.json.gz"
DATA = EXP / "v25_semantic_router/cloud_dataset/v25_semantic_router.npz"
V37 = EXP / "v37_online_equations"
V53 = EXP / "v53_active_equation"
OUT = EXP / "v55_pair_equation"
BASE_P = 1035
MAX_ROOTS = 8


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def feasible_values(matrix, rhs, vector):
    """Enumerate the integer TP deltas allowed by historical equations."""
    constraints = LinearConstraint(matrix, rhs, rhs)
    bounds = Bounds(np.zeros(len(vector)), np.ones(len(vector)))
    integrality = np.ones(len(vector), dtype=np.int8)
    low = milp(vector, integrality=integrality, bounds=bounds, constraints=constraints)
    high = milp(-vector, integrality=integrality, bounds=bounds, constraints=constraints)
    if not low.success or not high.success:
        return []
    values = []
    for value in range(int(round(low.fun)), int(round(-high.fun)) + 1):
        result = milp(
            np.zeros(len(vector)), integrality=integrality, bounds=bounds,
            constraints=LinearConstraint(
                np.vstack([matrix, vector]), np.r_[rhs, value], np.r_[rhs, value]
            ),
        )
        if result.success:
            values.append(value)
    return values


def query_stats(matrix, rhs, vector, probability, base_fixed):
    possible = feasible_values(matrix, rhs, vector)
    if len(possible) < 2:
        return None
    distribution = prior_delta_distribution(vector, probability, possible)
    outcomes = []
    for value in possible:
        lower, upper, _ = solve_fixed_labels(
            np.vstack([matrix, vector]), np.r_[rhs, value], len(vector)
        )
        fixed = int(np.sum(lower == upper))
        outcomes.append({
            "delta": int(value),
            "prior_probability": float(distribution[value]),
            "fixed": fixed,
            "new_fixed": fixed - base_fixed,
        })
    return {
        "possible": possible,
        "outcomes": outcomes,
        "expected_new_fixed": float(sum(row["prior_probability"] * row["new_fixed"] for row in outcomes)),
        "minimum_new_fixed": min(row["new_fixed"] for row in outcomes),
        "expected_delta_tp": float(sum(row["prior_probability"] * row["delta"] for row in outcomes)),
        "maximum_fixed": max(row["fixed"] for row in outcomes),
    }


def action_for(entry, position):
    key = entry["key"]
    return {
        "action_id": f"v55_pair_{position}_{'delete' if entry['selected'] else 'add'}",
        "order_id": key[0],
        "remove_rids": [key[1]] if entry["selected"] else [],
        "add_rids": [] if entry["selected"] else [key[1]],
        "source": "historical_equation_active_pair",
        "expected_gain": entry["probability"] if not entry["selected"] else 1 - entry["probability"],
        "evidence": {
            "calibrated_label_probability": entry["probability"],
            "selected_by_champion": entry["selected"],
        },
    }


def emit(selected, exact, order_ids, champion_roots, records_by_order, keys, matrix, rhs, probability, champion_nodes):
    actions = list(exact["actions"]) + selected["actions"]
    roots = apply_actions(champion_roots, records_by_order, actions)
    predictions = sum(len(rows) for rows in roots.values())
    if predictions != BASE_P:
        raise RuntimeError(f"expected P={BASE_P}, got {predictions}")
    path = OUT / "submissions/v55_pair_equation_probe.csv"
    write_submission(path, order_ids, roots)
    index = {key: i for i, key in enumerate(keys)}
    vector = np.zeros(len(keys), dtype=np.float64)
    for action in selected["actions"]:
        for rid in action["add_rids"]:
            vector[index[(action["order_id"], rid)]] += 1
        for rid in action["remove_rids"]:
            vector[index[(action["order_id"], rid)]] -= 1
    possible = selected["stats"]["possible"]
    score_possibilities = [
        {"query_delta_tp": value, "tp": 956 + value,
         "score": round(2 * (956 + value) / (TRUE_ROOTS + predictions), 9)}
        for value in possible
    ]
    outcome_files = {}
    for value in possible:
        lower, upper, _ = solve_fixed_labels(
            np.vstack([matrix, vector]), np.r_[rhs, value], len(keys)
        )
        fixed, exact_actions = corrections_from_fixed(keys, lower, upper, champion_nodes)
        tp, outcome_p, exact_score = checkpoint_math(exact_actions)
        # Keep the measured pair only if it improves on the exact-equation checkpoint.
        query_tp = 956 + value
        query_score = 2 * query_tp / (TRUE_ROOTS + predictions)
        checkpoint_actions = actions if query_score >= exact_score else exact_actions
        checkpoint_tp = query_tp if query_score >= exact_score else tp
        checkpoint_p = predictions if query_score >= exact_score else outcome_p
        checkpoint_score = max(query_score, exact_score)
        output = apply_actions(champion_roots, records_by_order, checkpoint_actions)
        if sum(len(rows) for rows in output.values()) != checkpoint_p:
            raise RuntimeError("outcome prediction count mismatch")
        outcome_path = OUT / "outcomes" / f"v55_checkpoint_delta_{value:+d}".replace("+", "p").replace("-", "m")
        outcome_path = outcome_path.with_suffix(".csv")
        write_submission(outcome_path, order_ids, output)
        outcome_files[str(value)] = {
            "path": str(outcome_path), "tp": checkpoint_tp,
            "predictions": checkpoint_p, "score": checkpoint_score,
            "sha256": sha256(outcome_path), "fixed_label_count": len(fixed),
        }
    manifest = {
        "probe_id": "v55_pair_equation_probe", "path": str(path),
        "baseline": str(CHAMPION), "fixed_base": exact["probe_id"],
        "candidate_actions": selected["actions"], "actions": actions,
        "predictions": predictions, "sha256": sha256(path),
        "score_possibilities": score_possibilities,
    }
    (OUT / "manifests/v55_pair_equation_probe.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest, outcome_files


def main():
    for directory in (OUT, OUT / "submissions", OUT / "manifests", OUT / "outcomes"):
        directory.mkdir(parents=True, exist_ok=True)
    order_ids, champion_roots = load_submission(CHAMPION)
    champion_nodes = {(oid, node["@rid"]) for oid, rows in champion_roots.items() for node in rows}
    keys, matrix, rhs, _ = build_system(collect_scored_submissions(), champion_nodes)
    key_index = {key: i for i, key in enumerate(keys)}
    fixed_report = read_json(V37 / "report.json")
    fixed_keys = {(row["order_id"], row["rid"]) for row in fixed_report["fixed_labels"]}
    previous = set()
    manifest_path = V53 / "manifests/v53_active_equation_probe.json"
    if (V53 / "online_result.json").exists() and manifest_path.exists():
        for action in read_json(manifest_path)["candidate_actions"]:
            previous |= {(action["order_id"], rid) for rid in action["remove_rids"] + action["add_rids"]}
    with gzip.open(RECORDS, "rt", encoding="utf-8") as handle:
        records = json.load(handle)["test"]
    records_by_order = {row["order_id"]: row for row in records}
    order_counts = {oid: len(nodes) for oid, nodes in champion_roots.items()}
    row_for_key, row = {}, 0
    for order in records:
        for alarm in order["alarms"]:
            row_for_key[(order["order_id"], alarm["rid"])] = row
            row += 1
    with np.load(DATA, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    probability_full = priors(arrays)["platt_station_domain_mean"]
    probability = np.asarray([probability_full[row_for_key[key]] for key in keys])
    eligible = []
    for i, key in enumerate(keys):
        if key in fixed_keys or key in previous:
            continue
        selected = key in champion_nodes
        if (selected and order_counts[key[0]] <= 1) or (not selected and order_counts[key[0]] >= MAX_ROOTS):
            continue
        eligible.append({"key": key, "index": i, "selected": selected, "probability": float(probability[i])})
    adds = [row for row in eligible if not row["selected"]]
    deletes = [row for row in eligible if row["selected"]]
    base_fixed = len(fixed_keys)
    screened = []
    matrix_rank = np.linalg.matrix_rank(matrix)
    for add in adds:
        for delete in deletes:
            if add["key"][0] == delete["key"][0]:
                continue
            vector = np.zeros(len(keys), dtype=np.float64)
            vector[add["index"]] = 1
            vector[delete["index"]] = -1
            rank_gain = int(np.linalg.matrix_rank(np.vstack([matrix, vector])) - matrix_rank)
            if not rank_gain:
                continue
            possible = feasible_values(matrix, rhs, vector)
            if len(possible) < 2:
                continue
            distribution = prior_delta_distribution(vector, probability, possible)
            expected_delta = sum(value * distribution[value] for value in possible)
            screened.append({"add": add, "delete": delete, "possible": possible,
                             "expected_delta_tp": float(expected_delta), "rank_gain": rank_gain})
    # Limit expensive binary-label bound solves to the pairs with the strongest
    # immediate expected gain and widest feasible response.
    screened.sort(key=lambda row: (-row["expected_delta_tp"], -len(row["possible"]),
                                   row["add"]["key"], row["delete"]["key"]))
    detailed = []
    for row in screened[:12]:
        vector = np.zeros(len(keys), dtype=np.float64)
        vector[row["add"]["index"]] = 1
        vector[row["delete"]["index"]] = -1
        stats = query_stats(matrix, rhs, vector, probability, base_fixed)
        if stats is None:
            continue
        actions = [action_for(row["delete"], 1), action_for(row["add"], 2)]
        detailed.append({**row, "actions": actions, "stats": stats})
    detailed.sort(key=lambda row: (-row["stats"]["expected_new_fixed"],
                                    -row["stats"]["expected_delta_tp"],
                                    -row["stats"]["maximum_fixed"],
                                    row["add"]["key"], row["delete"]["key"]))
    if not detailed:
        raise RuntimeError("no informative legal add/delete pairs found")
    selected = detailed[0]
    exact = read_json(V37 / "manifests/v37_exact_corrections.json")
    manifest, outcome_files = emit(selected, exact, order_ids, champion_roots, records_by_order,
                                   keys, matrix, rhs, probability, champion_nodes)
    report = {
        "version": "v55-pair-equation-1", "system": {
            "equations": len(matrix), "variables": len(keys), "rank": int(matrix_rank),
            "fixed": base_fixed,
        },
        "eligible": {"adds": len(adds), "deletes": len(deletes), "pairs_screened": len(screened),
                     "pairs_detailed": len(detailed)},
        "selection_rule": ["screen all legal pairs by expected TP delta", "within the top 12, maximize expected newly fixed labels", "maximize expected TP delta", "maximize resulting fixed labels"],
        "selected": selected, "top_pairs": detailed[:20], "probe": manifest,
        "outcome_files": outcome_files,
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "system": report["system"], "eligible": report["eligible"],
        "selected": {"add": selected["add"], "delete": selected["delete"], "stats": selected["stats"]},
        "probe": {key: manifest[key] for key in ("path", "predictions", "sha256", "score_possibilities")},
        "outcomes": outcome_files,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
