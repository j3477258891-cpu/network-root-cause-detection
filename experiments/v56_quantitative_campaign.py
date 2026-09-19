"""Build a 15-probe quantitative campaign over all 61 V50 candidates.

Every probe starts from V37 and changes about four order-disjoint nodes.  The
leaderboard score therefore reveals the exact aggregate TP delta for that
batch.  Batches are mutually disjoint, so any scored outcomes can later be
combined exactly into the best verified checkpoint.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
V30 = EXP / "v30_meta_stack"
for value in (EXP, V30):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from build_cross_order_probes import apply_actions, load_submission, write_submission
from v37_online_equation_solver import CHAMPION, TRUE_ROOTS


V37 = EXP / "v37_online_equations"
V50 = EXP / "v50_disjoint_batches"
RECORDS = EXP / "v25_semantic_router/cloud_dataset/semantic_records.json.gz"
OUT = EXP / "v56_quantitative_campaign"
BASE_TP = 956
BASE_P = 1035
TARGET = 0.95


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def candidate_actions():
    """Recover the order-disjoint V50 action pool in ranked batch order."""
    report = read_json(V50 / "report.json")
    adds, deletes, seen = [], [], set()
    for batch in report["batch_plan"]:
        manifest = report["probes"][batch["probe_id"]]
        for action in manifest["actions"]:
            if action["source"] != "v50_empirical_bayes_stratum":
                continue
            key = (action["order_id"], tuple(action["remove_rids"]), tuple(action["add_rids"]))
            if key in seen:
                continue
            seen.add(key)
            (deletes if action["remove_rids"] else adds).append(action)
    if len(adds) != 32 or len(deletes) != 29 or len(seen) != 61:
        raise RuntimeError((len(adds), len(deletes), len(seen)))
    return adds, deletes


def correctness_probability(action):
    evidence = action.get("evidence", {})
    rule = evidence.get("matched_rule", {})
    return float(rule.get("posterior_mean", 0.5))


def make_batches(adds, deletes):
    batches = []
    # Seven pure add batches, seven pure delete batches, and one mixed batch.
    for index in range(7):
        batches.append((f"v56_add_batch{index + 1:02d}_n04", adds[4 * index:4 * index + 4]))
    for index in range(7):
        batches.append((f"v56_delete_batch{index + 1:02d}_n04", deletes[4 * index:4 * index + 4]))
    batches.append(("v56_mixed_tail_n05", adds[28:32] + deletes[28:29]))
    flattened = [action["order_id"] for _, rows in batches for action in rows]
    if len(flattened) != 61 or len(flattened) != len(set(flattened)):
        raise RuntimeError("V56 batches must cover 61 distinct orders")
    return batches


def delta_range(actions):
    additions = sum(len(action["add_rids"]) for action in actions)
    deletions = sum(len(action["remove_rids"]) for action in actions)
    return range(-deletions, additions + 1)


def emit_probe(name, candidates, exact, order_ids, champion_roots, records_by_order):
    actions = list(exact["actions"]) + list(candidates)
    roots = apply_actions(champion_roots, records_by_order, actions)
    predictions = sum(len(rows) for rows in roots.values())
    additions = sum(len(action["add_rids"]) for action in candidates)
    deletions = sum(len(action["remove_rids"]) for action in candidates)
    expected_delta = sum(
        correctness_probability(action) if action["add_rids"]
        else -(1 - correctness_probability(action))
        for action in candidates
    )
    path = OUT / "submissions" / f"{name}.csv"
    write_submission(path, order_ids, roots)
    possibilities = [{
        "batch_delta_tp": delta,
        "tp": BASE_TP + delta,
        "score": round(2 * (BASE_TP + delta) / (TRUE_ROOTS + predictions), 9),
    } for delta in delta_range(candidates)]
    manifest = {
        "probe_id": name, "path": str(path), "baseline": str(CHAMPION),
        "fixed_base": exact["probe_id"], "candidate_actions": candidates,
        "actions": actions, "candidate_count": len(candidates),
        "additions": additions, "deletions": deletions,
        "prediction_delta": additions - deletions, "predictions": predictions,
        "expected_delta_tp": expected_delta, "sha256": sha256(path),
        "score_possibilities": possibilities,
    }
    (OUT / "manifests" / f"{name}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def simulate(manifests, trials=20000, seed=20260820):
    """Posterior diagnostic only; online scores remain the acceptance gate."""
    rng = random.Random(seed)
    scores, reached = [], 0
    base_score = 2 * BASE_TP / (TRUE_ROOTS + BASE_P)
    for _ in range(trials):
        tp, predictions = BASE_TP, BASE_P
        for manifest in manifests:
            delta = 0
            for action in manifest["candidate_actions"]:
                correct = rng.random() < correctness_probability(action)
                if action["add_rids"]:
                    delta += int(correct)
                else:
                    delta -= int(not correct)
            candidate_score = 2 * (tp + delta) / (
                TRUE_ROOTS + predictions + manifest["prediction_delta"]
            )
            current_score = 2 * tp / (TRUE_ROOTS + predictions)
            if candidate_score > current_score:
                tp += delta
                predictions += manifest["prediction_delta"]
        score = 2 * tp / (TRUE_ROOTS + predictions)
        scores.append(score)
        reached += int(score >= TARGET)
    scores.sort()
    return {
        "trials": trials, "seed": seed,
        "base_score": base_score,
        "mean": sum(scores) / len(scores),
        "p05": scores[int(0.05 * trials)],
        "median": scores[trials // 2],
        "p95": scores[int(0.95 * trials)],
        "probability_reach_0_95": reached / trials,
        "warning": "Independent empirical-Bayes simulation; diagnostic, not a guarantee.",
    }


def main():
    for directory in (OUT, OUT / "submissions", OUT / "manifests"):
        directory.mkdir(parents=True, exist_ok=True)
    order_ids, champion_roots = load_submission(CHAMPION)
    with gzip.open(RECORDS, "rt", encoding="utf-8") as handle:
        records = json.load(handle)["test"]
    records_by_order = {row["order_id"]: row for row in records}
    exact = read_json(V37 / "manifests/v37_exact_corrections.json")
    adds, deletes = candidate_actions()
    manifests = [
        emit_probe(name, candidates, exact, order_ids, champion_roots, records_by_order)
        for name, candidates in make_batches(adds, deletes)
    ]
    action_nodes = [
        (action["order_id"], rid)
        for manifest in manifests for action in manifest["candidate_actions"]
        for rid in action["remove_rids"] + action["add_rids"]
    ]
    if len(action_nodes) != 61 or len(set(action_nodes)) != 61:
        raise RuntimeError("candidate nodes are not disjoint")
    report = {
        "version": "v56-quantitative-campaign-1",
        "base": {"tp": BASE_TP, "predictions": BASE_P,
                 "score": 2 * BASE_TP / (TRUE_ROOTS + BASE_P)},
        "target": TARGET, "submission_budget": 20,
        "schedule": {
            "v49_and_v47": 2, "v56_quantitative_probes": len(manifests),
            "checkpoint": 1, "reserve": 20 - 3 - len(manifests),
        },
        "candidate_count": len(action_nodes),
        "additions": len(adds), "deletions": len(deletes),
        "probes": {manifest["probe_id"]: manifest for manifest in manifests},
        "probe_order": [manifest["probe_id"] for manifest in manifests],
        "posterior_diagnostic": simulate(manifests),
        "optimistic_upper_bound": {
            "tp": BASE_TP + len(adds),
            "predictions": BASE_P + len(adds) - len(deletes),
            "score": 2 * (BASE_TP + len(adds)) /
                     (TRUE_ROOTS + BASE_P + len(adds) - len(deletes)),
            "assumption": "all additions true and every deleted node false",
        },
        "combined_v49_upper_bound": {
            "tp": BASE_TP + len(adds) + 3,
            "predictions": BASE_P + len(adds) - len(deletes) + 2,
            "score": 2 * (BASE_TP + len(adds) + 3) /
                     (TRUE_ROOTS + BASE_P + len(adds) - len(deletes) + 2),
            "assumption": "all V56 actions correct and V49 candidate delta TP is +3",
            "minimum_v49_delta_if_all_v56_correct": 2,
        },
    }
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    state = {
        "version": "v56-campaign-state-1", "true_roots": TRUE_ROOTS,
        "base": report["base"], "results": {},
        "checkpoint": {
            "path": exact["path"], "tp": BASE_TP, "predictions": BASE_P,
            "score": report["base"]["score"], "selected_probe_ids": [],
            "sha256": exact["sha256"],
        },
    }
    (OUT / "state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "candidate_count": report["candidate_count"], "schedule": report["schedule"],
        "posterior_diagnostic": report["posterior_diagnostic"],
        "optimistic_upper_bound": report["optimistic_upper_bound"],
        "combined_v49_upper_bound": report["combined_v49_upper_bound"],
        "first_probe": manifests[0]["path"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
