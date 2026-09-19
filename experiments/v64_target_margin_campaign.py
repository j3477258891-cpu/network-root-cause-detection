"""Build a 20-query disjoint batch campaign optimized for F1 >= 0.95.

Each query contains a disjoint group of either additions or deletions relative
to the V37 exact base.  The public score reveals the exact number of correct
actions in that group.  Because groups are disjoint, their TP and prediction
deltas add exactly and any scored subset can be combined without another
probe.  Groups are selected by dynamic programming to maximize expected
positive margin at the target F1, rather than to recover every action label.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
V30 = EXP / "v30_meta_stack"
for value in (EXP, V30):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from build_cross_order_probes import apply_actions, load_submission, write_submission
from v37_online_equation_solver import CHAMPION, TRUE_ROOTS
from v58_coded_campaign import action_key, sha256


V37 = EXP / "v37_online_equations"
V58 = EXP / "v58_coded_campaign"
V59 = EXP / "v59_extended_coded_campaign"
RECORDS = EXP / "v25_semantic_router/cloud_dataset/semantic_records.json.gz"
OUT = EXP / "v64_target_margin_campaign"
BASE_TP = 956
BASE_P = 1035
TARGET = 0.95
QUERY_BUDGET = 20
SIMULATION_TRIALS = 100_000


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def candidate_pool():
    reports = [read_json(V58 / "report.json"), read_json(V59 / "report.json")]
    actions = reports[0]["candidate_actions"] + reports[1]["candidate_actions"]
    probabilities = np.asarray(
        reports[0]["correctness_probabilities"]
        + reports[1]["correctness_probabilities"],
        dtype=np.float64,
    )
    if len(actions) != 290 or len({action_key(action) for action in actions}) != 290:
        raise RuntimeError("expected 290 unique candidate actions")
    orders = [action["order_id"] for action in actions]
    if len(set(orders)) != len(orders):
        raise RuntimeError("candidate actions are not order-disjoint")
    return actions, probabilities


def positive_margin_distribution(probabilities, kind):
    """Return expected positive target margin and its exact count PMF."""
    pmf = np.array([1.0], dtype=np.float64)
    for probability in probabilities:
        pmf = np.convolve(pmf, [1 - probability, probability])
    size = len(probabilities)
    counts = np.arange(size + 1, dtype=np.float64)
    prediction_delta = size if kind == "add" else -size
    tp_delta = counts if kind == "add" else counts - size
    margins = 2 * tp_delta - TARGET * prediction_delta
    expected_positive = float(pmf @ np.maximum(margins, 0.0))
    return expected_positive, pmf, margins


def segment_utilities(probabilities, kind):
    """Compute expected positive target margin for every contiguous segment."""
    count = len(probabilities)
    utility = np.full((count, count + 1), -np.inf, dtype=np.float64)
    for start in range(count):
        pmf = np.array([1.0], dtype=np.float64)
        for stop in range(start + 1, count + 1):
            probability = probabilities[stop - 1]
            pmf = np.convolve(pmf, [1 - probability, probability])
            size = stop - start
            counts = np.arange(size + 1, dtype=np.float64)
            prediction_delta = size if kind == "add" else -size
            tp_delta = counts if kind == "add" else counts - size
            margins = 2 * tp_delta - TARGET * prediction_delta
            utility[start, stop] = float(pmf @ np.maximum(margins, 0.0))
    return utility


def best_prefix_partition(probabilities, kind, max_groups):
    """Best partition of any sorted prefix for every group count."""
    utility = segment_utilities(probabilities, kind)
    count = len(probabilities)
    dp = np.full((max_groups + 1, count + 1), -np.inf, dtype=np.float64)
    parent = np.full((max_groups + 1, count + 1), -1, dtype=np.int32)
    dp[0, 0] = 0.0
    for groups in range(1, max_groups + 1):
        for stop in range(groups, count + 1):
            starts = np.arange(groups - 1, stop)
            values = dp[groups - 1, starts] + utility[starts, stop]
            best_offset = int(np.argmax(values))
            dp[groups, stop] = values[best_offset]
            parent[groups, stop] = int(starts[best_offset])
    results = {}
    for groups in range(1, max_groups + 1):
        stop = int(np.argmax(dp[groups]))
        segments = []
        cursor = stop
        for level in range(groups, 0, -1):
            start = int(parent[level, cursor])
            segments.append((start, cursor))
            cursor = start
        segments.reverse()
        results[groups] = {
            "expected_positive_margin": float(dp[groups, stop]),
            "prefix_count": stop,
            "segments": segments,
        }
    return results


def optimize_groups(actions, probabilities):
    by_kind = {}
    partitions = {}
    for kind in ("add", "delete"):
        indices = [
            index for index, action in enumerate(actions)
            if bool(action["add_rids"]) == (kind == "add")
        ]
        indices.sort(key=lambda index: (-probabilities[index], index))
        by_kind[kind] = indices
        partitions[kind] = best_prefix_partition(
            probabilities[indices], kind, QUERY_BUDGET - 1
        )

    allocations = []
    for add_groups in range(1, QUERY_BUDGET):
        delete_groups = QUERY_BUDGET - add_groups
        add = partitions["add"][add_groups]
        delete = partitions["delete"][delete_groups]
        allocations.append({
            "add_groups": add_groups,
            "delete_groups": delete_groups,
            "expected_positive_margin": (
                add["expected_positive_margin"]
                + delete["expected_positive_margin"]
            ),
        })
    allocation = max(allocations, key=lambda row: row["expected_positive_margin"])

    groups = []
    for kind, group_count in (
        ("add", allocation["add_groups"]),
        ("delete", allocation["delete_groups"]),
    ):
        partition = partitions[kind][group_count]
        ordered = by_kind[kind]
        for start, stop in partition["segments"]:
            candidate_indices = ordered[start:stop]
            expected_positive, pmf, margins = positive_margin_distribution(
                probabilities[candidate_indices], kind
            )
            groups.append({
                "kind": kind,
                "candidate_indices": candidate_indices,
                "expected_positive_margin": expected_positive,
                "mean_correct": float(probabilities[candidate_indices].sum()),
                "min_probability": float(probabilities[candidate_indices].min()),
                "max_probability": float(probabilities[candidate_indices].max()),
                "positive_count_probability": float(pmf[margins > 0].sum()),
            })
    groups.sort(key=lambda row: row["expected_positive_margin"], reverse=True)
    for index, group in enumerate(groups, 1):
        group["probe_id"] = f"v64_margin_{index:02d}"
    return groups, allocation, allocations


def emit_probe(group, actions, exact, order_ids, champion_roots, records_by_order):
    selected = [dict(actions[index]) for index in group["candidate_indices"]]
    all_actions = list(exact["actions"]) + selected
    roots = apply_actions(champion_roots, records_by_order, all_actions)
    additions = sum(bool(action["add_rids"]) for action in selected)
    deletions = len(selected) - additions
    predictions = sum(len(values) for values in roots.values())
    if predictions != BASE_P + additions - deletions:
        raise RuntimeError("V64 prediction count mismatch")
    path = OUT / "submissions" / f"{group['probe_id']}.csv"
    write_submission(path, order_ids, roots)
    possibilities = []
    for correct_count in range(len(selected) + 1):
        tp_delta = correct_count - deletions
        tp = BASE_TP + tp_delta
        score = 2 * tp / (TRUE_ROOTS + predictions)
        possibilities.append({
            "correct_count": correct_count,
            "tp_delta": tp_delta,
            "prediction_delta": additions - deletions,
            "target_margin": 2 * tp_delta - TARGET * (additions - deletions),
            "tp": tp,
            "score": round(score, 9),
        })
    manifest = {
        **group,
        "path": str(path),
        "sha256": sha256(path),
        "actions": all_actions,
        "candidate_actions": selected,
        "additions": additions,
        "deletions": deletions,
        "predictions": predictions,
        "prediction_delta": additions - deletions,
        "score_possibilities": possibilities,
    }
    (OUT / "manifests" / f"{group['probe_id']}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def simulate(groups, probabilities, trials=SIMULATION_TRIALS, seed=20260827):
    rng = np.random.default_rng(seed)
    scores = np.empty(trials, dtype=np.float64)
    margins = np.empty(trials, dtype=np.float64)
    for trial in range(trials):
        labels = rng.random(len(probabilities)) < probabilities
        tp_delta = 0
        prediction_delta = 0
        positive_margin = 0.0
        for group in groups:
            indices = group["candidate_indices"]
            size = len(indices)
            correct = int(labels[indices].sum())
            group_tp_delta = correct if group["kind"] == "add" else correct - size
            group_prediction_delta = size if group["kind"] == "add" else -size
            margin = 2 * group_tp_delta - TARGET * group_prediction_delta
            if margin > 0:
                tp_delta += group_tp_delta
                prediction_delta += group_prediction_delta
                positive_margin += margin
        scores[trial] = 2 * (BASE_TP + tp_delta) / (
            TRUE_ROOTS + BASE_P + prediction_delta
        )
        margins[trial] = positive_margin
    return {
        "trials": trials,
        "mean": float(scores.mean()),
        "p01": float(np.quantile(scores, 0.01)),
        "p05": float(np.quantile(scores, 0.05)),
        "median": float(np.median(scores)),
        "p95": float(np.quantile(scores, 0.95)),
        "probability_reach_0_94": float(np.mean(scores >= 0.94)),
        "probability_reach_0_95": float(np.mean(scores >= TARGET)),
        "mean_positive_target_margin": float(margins.mean()),
        "base_target_margin_deficit": float(
            TARGET * (TRUE_ROOTS + BASE_P) - 2 * BASE_TP
        ),
        "warning": "Independent Bernoulli simulation using offline-calibrated probabilities.",
    }


def main():
    for directory in (OUT, OUT / "submissions", OUT / "manifests"):
        directory.mkdir(parents=True, exist_ok=True)
    actions, probabilities = candidate_pool()
    groups, allocation, allocation_table = optimize_groups(actions, probabilities)
    if len(groups) != QUERY_BUDGET:
        raise RuntimeError("wrong V64 query count")
    selected_indices = [index for group in groups for index in group["candidate_indices"]]
    if len(selected_indices) != len(set(selected_indices)):
        raise RuntimeError("V64 groups overlap")

    order_ids, champion_roots = load_submission(CHAMPION)
    with gzip.open(RECORDS, "rt", encoding="utf-8") as handle:
        records = json.load(handle)["test"]
    records_by_order = {row["order_id"]: row for row in records}
    exact = read_json(V37 / "manifests/v37_exact_corrections.json")
    manifests = [
        emit_probe(group, actions, exact, order_ids, champion_roots, records_by_order)
        for group in groups
    ]
    simulation = simulate(groups, probabilities)
    report = {
        "version": "v64-target-margin-campaign-1",
        "target": TARGET,
        "base": {
            "tp": BASE_TP,
            "predictions": BASE_P,
            "score": 2 * BASE_TP / (TRUE_ROOTS + BASE_P),
        },
        "candidate_pool_count": len(actions),
        "selected_candidate_count": len(selected_indices),
        "query_count": len(groups),
        "final_checkpoint_queries": 1,
        "estimated_days_at_two_submissions": (len(groups) + 1) / 2,
        "allocation": allocation,
        "allocation_table": allocation_table,
        "groups": manifests,
        "simulation": simulation,
        "invariant": "All groups use disjoint orders, so observed TP and P deltas add exactly.",
    }
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    state = {
        "version": "v64-target-margin-state-1",
        "results": {},
        "accepted_probe_ids": [],
        "checkpoint": {
            "path": str(V37 / "submissions/v37_exact_corrections.csv"),
            "tp": BASE_TP,
            "predictions": BASE_P,
            "score": 2 * BASE_TP / (TRUE_ROOTS + BASE_P),
        },
        "target_reached": False,
        "next_probe": {
            key: manifests[0][key] for key in ("probe_id", "path", "sha256", "predictions")
        },
    }
    (OUT / "state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "query_count": len(groups),
        "selected_candidate_count": len(selected_indices),
        "allocation": allocation,
        "simulation": simulation,
        "first_probe": state["next_probe"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
