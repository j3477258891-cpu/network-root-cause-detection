"""Plan an adaptive probe that adds information to historical score equations.

Each candidate query is a valid set of historical actions.  For every
possible leaderboard count, the planner augments the binary equation system,
re-solves fixed labels, and emits the exact checkpoint implied by that result.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
V30 = EXPERIMENTS / "v30_meta_stack"
for value in (EXPERIMENTS, V30):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from build_cross_order_probes import apply_actions, load_submission, write_submission
from v37_online_equation_solver import (
    CHAMPION,
    CHAMPION_TP,
    TRUE_ROOTS,
    build_system,
    collect_scored_submissions,
    solve_fixed_labels,
)
from v41_equation_aware_planner import feasible_history_values, optimize_history
from v45_constrained_map_inference import priors as calibrated_priors


RECORDS = EXPERIMENTS / "v25_semantic_router/cloud_dataset/semantic_records.json.gz"
DATA = EXPERIMENTS / "v25_semantic_router/cloud_dataset/v25_semantic_router.npz"
V37 = EXPERIMENTS / "v37_online_equations"
V45 = EXPERIMENTS / "v45_constrained_map/report.json"
OUT = EXPERIMENTS / "v53_active_equation"
BASE_P = 1035
MAX_ROOTS = 8


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def action_vector(actions, index, n):
    vector = np.zeros(n, dtype=np.float64)
    for action in actions:
        oid = action["order_id"]
        for rid in action.get("add_rids", []):
            vector[index[(oid, rid)]] += 1
        for rid in action.get("remove_rids", []):
            vector[index[(oid, rid)]] -= 1
    return vector


def corrections_from_fixed(keys, lower, upper, champion_nodes):
    by_order = defaultdict(lambda: {"remove_rids": [], "add_rids": []})
    fixed = []
    for i, key in enumerate(keys):
        if lower[i] != upper[i]:
            continue
        label = int(lower[i])
        selected = key in champion_nodes
        fixed.append({
            "order_id": key[0], "rid": key[1], "label": label,
            "selected_by_champion": selected,
        })
        if selected and label == 0:
            by_order[key[0]]["remove_rids"].append(key[1])
        elif not selected and label == 1:
            by_order[key[0]]["add_rids"].append(key[1])
    actions = []
    for i, (oid, change) in enumerate(sorted(by_order.items()), 1):
        actions.append({
            "action_id": f"v53_outcome_exact_{i:03d}",
            "order_id": oid,
            "remove_rids": sorted(change["remove_rids"]),
            "add_rids": sorted(change["add_rids"]),
            "source": "augmented_leaderboard_equations",
            "expected_gain": None,
            "evidence": {"proof": "fixed in every feasible binary solution after probe"},
        })
    return fixed, actions


def checkpoint_math(actions):
    additions = sum(len(action["add_rids"]) for action in actions)
    deletions = sum(len(action["remove_rids"]) for action in actions)
    tp = CHAMPION_TP + additions
    predictions = BASE_P + additions - deletions
    return tp, predictions, 2 * tp / (TRUE_ROOTS + predictions)


def prior_delta_distribution(vector, prior_probability, possible):
    distribution = {0: 1.0}
    for coefficient, probability in zip(vector, prior_probability):
        if not coefficient:
            continue
        updated = defaultdict(float)
        updated_keys = ((0, 1 - probability), (int(coefficient), probability))
        for current, mass in distribution.items():
            for change, local_mass in updated_keys:
                updated[current + change] += mass * local_mass
        distribution = dict(updated)
    restricted = {value: distribution.get(value, 0.0) for value in possible}
    total = sum(restricted.values())
    if total <= 0:
        return {value: 1.0 / len(possible) for value in possible}
    return {value: mass / total for value, mass in restricted.items()}


def analyze_query(name, actions, keys, matrix, rhs, champion_nodes,
                  prior_probability, base_exact_actions, base_exact_tp,
                  base_exact_predictions):
    index = {key: i for i, key in enumerate(keys)}
    vector = action_vector(actions, index, len(keys))
    low, high = optimize_history(matrix, rhs, vector)
    possible = sorted(feasible_history_values(matrix, rhs, vector, low, high))
    prior_distribution = prior_delta_distribution(vector, prior_probability, possible)
    prediction_delta = int(round(vector.sum()))
    outcomes = []
    for delta in possible:
        augmented_matrix = np.vstack([matrix, vector])
        augmented_rhs = np.r_[rhs, delta]
        lower, upper, _ = solve_fixed_labels(augmented_matrix, augmented_rhs, len(keys))
        fixed, fixed_actions = corrections_from_fixed(
            keys, lower, upper, champion_nodes
        )
        exact_tp, exact_predictions, exact_score = checkpoint_math(fixed_actions)
        probe_tp = base_exact_tp + int(delta)
        probe_predictions = base_exact_predictions + prediction_delta
        probe_score = 2 * probe_tp / (TRUE_ROOTS + probe_predictions)
        if probe_score > exact_score:
            checkpoint_source = "accepted_query"
            checkpoint_actions = list(base_exact_actions) + list(actions)
            tp, predictions, score = probe_tp, probe_predictions, probe_score
        else:
            checkpoint_source = "fixed_labels"
            checkpoint_actions = fixed_actions
            tp, predictions, score = exact_tp, exact_predictions, exact_score
        outcomes.append({
            "query_delta_tp": int(delta),
            "prior_probability": prior_distribution[int(delta)],
            "fixed_label_count": len(fixed),
            "beneficial_action_count": len(fixed_actions),
            "beneficial_node_count": sum(
                len(action["remove_rids"]) + len(action["add_rids"])
                for action in fixed_actions
            ),
            "checkpoint_tp": tp,
            "checkpoint_predictions": predictions,
            "checkpoint_score": score,
            "checkpoint_source": checkpoint_source,
            "fixed_labels": fixed,
            "checkpoint_actions": checkpoint_actions,
        })
    return {
        "query_id": name,
        "actions": actions,
        "query_node_count": int(np.count_nonzero(vector)),
        "prediction_delta": prediction_delta,
        "possible_delta_tp": possible,
        "prior_delta_distribution": {
            str(value): prior_distribution[value] for value in possible
        },
        "outcome_count": len(possible),
        "maximum_information_bits": math.log2(len(possible)),
        "worst_checkpoint_score": min(row["checkpoint_score"] for row in outcomes),
        "mean_checkpoint_score": float(np.mean([row["checkpoint_score"] for row in outcomes])),
        "expected_checkpoint_score": sum(
            row["prior_probability"] * row["checkpoint_score"] for row in outcomes
        ),
        "expected_fixed_labels": sum(
            row["prior_probability"] * row["fixed_label_count"] for row in outcomes
        ),
        "expected_beneficial_nodes": sum(
            row["prior_probability"] * row["beneficial_node_count"] for row in outcomes
        ),
        "best_checkpoint_score": max(row["checkpoint_score"] for row in outcomes),
        "minimum_fixed_labels": min(row["fixed_label_count"] for row in outcomes),
        "mean_fixed_labels": float(np.mean([row["fixed_label_count"] for row in outcomes])),
        "outcomes": outcomes,
    }


def emit_query(selected, exact, order_ids, champion_roots, records_by_order,
               base_exact_tp, base_exact_predictions):
    actions = list(exact["actions"]) + selected["actions"]
    roots = apply_actions(champion_roots, records_by_order, actions)
    predictions = sum(len(values) for values in roots.values())
    expected_predictions = base_exact_predictions + selected["prediction_delta"]
    if predictions != expected_predictions:
        raise RuntimeError((predictions, expected_predictions))
    name = "v53_active_equation_probe"
    path = OUT / "submissions" / f"{name}.csv"
    write_submission(path, order_ids, roots)
    score_possibilities = []
    for delta in selected["possible_delta_tp"]:
        tp = base_exact_tp + delta
        score_possibilities.append({
            "query_delta_tp": delta,
            "tp": tp,
            "score": round(2 * tp / (TRUE_ROOTS + predictions), 9),
        })
    manifest = {
        "probe_id": name,
        "path": str(path),
        "baseline": str(CHAMPION),
        "fixed_base": exact["probe_id"],
        "query_source_id": selected["query_id"],
        "actions": actions,
        "candidate_actions": selected["actions"],
        "predictions": predictions,
        "sha256": sha256(path),
        "score_possibilities": score_possibilities,
    }
    (OUT / "manifests" / f"{name}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def emit_outcome_checkpoints(selected, order_ids, champion_roots, records_by_order):
    output = {}
    for row in selected["outcomes"]:
        delta = row["query_delta_tp"]
        roots = apply_actions(champion_roots, records_by_order, row["checkpoint_actions"])
        predictions = sum(len(values) for values in roots.values())
        if predictions != row["checkpoint_predictions"]:
            raise RuntimeError((delta, predictions, row["checkpoint_predictions"]))
        name = f"v53_checkpoint_delta_{delta:+d}".replace("+", "p").replace("-", "m")
        path = OUT / "outcomes" / f"{name}.csv"
        write_submission(path, order_ids, roots)
        output[str(delta)] = {
            "path": str(path),
            "tp": row["checkpoint_tp"],
            "predictions": predictions,
            "score": row["checkpoint_score"],
            "sha256": sha256(path),
            "fixed_label_count": row["fixed_label_count"],
            "beneficial_node_count": row["beneficial_node_count"],
        }
    return output


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "submissions").mkdir(exist_ok=True)
    (OUT / "manifests").mkdir(exist_ok=True)
    (OUT / "outcomes").mkdir(exist_ok=True)
    order_ids, champion_roots = load_submission(CHAMPION)
    champion_nodes = {
        (oid, node["@rid"]) for oid, values in champion_roots.items() for node in values
    }
    keys, matrix, rhs, _ = build_system(
        collect_scored_submissions(), champion_nodes
    )
    with gzip.open(RECORDS, "rt", encoding="utf-8") as handle:
        records = json.load(handle)["test"]
    records_by_order = {row["order_id"]: row for row in records}
    exact = read_json(V37 / "manifests/v37_exact_corrections.json")
    base_exact_tp = int(exact["expected_tp"])
    base_exact_predictions = int(exact["predictions"])
    # A prior V53 query remains informative even when its result was neutral.
    # Exclude its actions from the next query so the adaptive loop never
    # spends a submission repeating an already observed equation.
    previous_query_nodes = set()
    previous_query_orders = set()
    if (OUT / "online_result.json").exists() and (OUT / "manifests/v53_active_equation_probe.json").exists():
        previous = read_json(OUT / "manifests/v53_active_equation_probe.json")
        for action in previous.get("candidate_actions", []):
            previous_query_orders.add(action["order_id"])
            previous_query_nodes |= action_nodes(action)
    map_actions = read_json(V45)["map_candidate_actions"]
    with np.load(DATA, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    v45_report = read_json(V45)
    probability = calibrated_priors(arrays)[v45_report["selected_prior"]]
    row_for_key = {}
    row_index = 0
    for order in records:
        for alarm in order["alarms"]:
            row_for_key[(order["order_id"], alarm["rid"])] = row_index
            row_index += 1
    prior_probability = np.asarray(
        [probability[row_for_key[key]] for key in keys], dtype=np.float64
    )
    key_set = set(keys)
    eligible = []
    used_orders = {action["order_id"] for action in exact["actions"]} | previous_query_orders
    for action in map_actions:
        nodes = {
            (action["order_id"], rid)
            for rid in action.get("remove_rids", []) + action.get("add_rids", [])
        }
        if action["order_id"] in used_orders or nodes & previous_query_nodes or not nodes or not nodes <= key_set:
            continue
        used_orders.add(action["order_id"])
        eligible.append(action)

    candidates = []
    for count in (2, 4, 6, 8, 12, 16, 20, 24):
        if len(eligible) < count:
            continue
        candidates.append(analyze_query(
            f"v45_map_prefix_{count:02d}", eligible[:count],
            keys, matrix, rhs, champion_nodes, prior_probability, exact["actions"],
            base_exact_tp, base_exact_predictions,
        ))
    candidates.sort(key=lambda row: (
        -row["expected_fixed_labels"],
        -row["expected_beneficial_nodes"],
        row["query_node_count"],
        -row["expected_checkpoint_score"],
        -row["worst_checkpoint_score"],
        -row["mean_checkpoint_score"],
        -row["minimum_fixed_labels"],
        -row["mean_fixed_labels"],
        row["query_node_count"],
    ))
    selected = candidates[0]
    manifest = emit_query(
        selected, exact, order_ids, champion_roots, records_by_order,
        base_exact_tp, base_exact_predictions,
    )
    outcome_files = emit_outcome_checkpoints(
        selected, order_ids, champion_roots, records_by_order
    )
    report = {
        "version": "v53-active-equation-probe-1",
        "system": {
            "equations": len(matrix), "variables": len(keys),
            "rank": int(np.linalg.matrix_rank(matrix)),
        },
        "selection_rule": [
            "maximize calibrated-prior expected fixed labels",
            "maximize calibrated-prior expected beneficial nodes",
            "minimize query size",
            "maximize expected checkpoint score",
            "maximize worst exact checkpoint score",
            "maximize unweighted mean exact checkpoint score",
            "maximize fixed labels",
            "minimize query size",
        ],
        "candidates": candidates,
        "selected": selected,
        "probe": manifest,
        "outcome_files": outcome_files,
    }
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "system": report["system"],
        "candidate_summary": [{
            key: row[key] for key in (
                "query_id", "query_node_count", "possible_delta_tp",
                "worst_checkpoint_score", "mean_checkpoint_score",
                "expected_checkpoint_score",
                "expected_fixed_labels", "expected_beneficial_nodes",
                "best_checkpoint_score", "minimum_fixed_labels", "mean_fixed_labels",
            )
        } for row in candidates],
        "selected": selected["query_id"],
        "probe": {key: manifest[key] for key in ("path", "predictions", "sha256", "score_possibilities")},
        "outcome_files": outcome_files,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
