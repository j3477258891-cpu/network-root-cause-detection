"""Build disjoint quantitative leaderboard batches beyond the V49 action set."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / ".deps"
V30 = ROOT / "experiments/v30_meta_stack"
EXPERIMENTS = ROOT / "experiments"
for value in (DEPS, V30, EXPERIMENTS):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from scipy.stats import beta as beta_distribution

from build_cross_order_probes import apply_actions, load_submission, write_submission
from v30_meta_stack import exact_count_mask
from v37_online_equation_solver import build_system, collect_scored_submissions
from v40_one_sided_boundary import rank_average, score_arrays, selected_mask, unique_boundary
from v48_stable_strata_batches import candidate_table, protected_nodes, rule_key


DATA = EXPERIMENTS / "v25_semantic_router/cloud_dataset/v25_semantic_router.npz"
RECORDS = EXPERIMENTS / "v25_semantic_router/cloud_dataset/semantic_records.json.gz"
CHAMPION = V30 / "submissions/v30_cross_order_top5.csv"
V37 = EXPERIMENTS / "v37_online_equations"
V49 = EXPERIMENTS / "v49_extended_count_batch"
OUT = EXPERIMENTS / "v50_disjoint_batches"
TRAIN_BUDGET = 3103
TRUE_ROOTS = 1044
BASE_TP = 956
BASE_P = 1035
BATCH_SIZE = 8
MAX_BATCHES_PER_KIND = 6


RULE_FIELDS = (
    ("rank_bin",), ("title",), ("device_type",), ("vendor",),
    ("deployment",), ("title", "rank_bin"),
    ("title", "device_type"), ("title", "vendor"),
    ("device_type", "rank_bin"), ("vendor", "rank_bin"),
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def root_nodes(roots):
    return {(oid, node["@rid"]) for oid, values in roots.items() for node in values}


def exploratory_rules(rows, threshold):
    grouped = defaultdict(list)
    for row in rows:
        for fields in RULE_FIELDS:
            key = rule_key(row, fields)
            if key is not None:
                grouped[key].append(row)
    rules = {}
    prior_strength = 10.0
    for key, members in grouped.items():
        support = len(members)
        fold_values = sorted({row["fold"] for row in members})
        if support < 5 or len(fold_values) < 2:
            continue
        correct = sum(row["target"] for row in members)
        precision = correct / support
        fold_precision = {
            str(fold): float(np.mean([row["target"] for row in members if row["fold"] == fold]))
            for fold in fold_values
        }
        posterior = (correct + prior_strength * threshold) / (support + prior_strength)
        lower75 = float(beta_distribution.ppf(
            0.25, 1 + correct + prior_strength * threshold,
            1 + support - correct + prior_strength * (1 - threshold),
        ))
        if precision < threshold:
            continue
        if posterior <= threshold:
            continue
        if lower75 < threshold - 0.10:
            continue
        rules[key] = {
            "fields": list(key[0]), "values": list(key[1]),
            "support": support, "correct": correct, "precision": precision,
            "fold_count": len(fold_values), "fold_precision": fold_precision,
            "posterior_mean": posterior, "beta_lower_75": lower75,
            "advantage": posterior - threshold,
        }
    return rules


def best_rule(row, rules):
    matches = []
    for fields in RULE_FIELDS:
        key = rule_key(row, fields)
        if key in rules:
            matches.append(rules[key])
    return max(matches, key=lambda rule: (
        rule["advantage"], rule["support"], len(rule["fields"])
    )) if matches else None


def select_candidates(rows, rules, excluded_nodes, excluded_orders):
    output = []
    for row in rows:
        if (row["order_id"], row["rid"]) in excluded_nodes:
            continue
        if row["order_id"] in excluded_orders:
            continue
        rule = best_rule(row, rules)
        if rule is None:
            continue
        output.append({**row, "matched_rule": rule, "estimated_advantage": rule["advantage"]})
    output.sort(key=lambda row: (-row["estimated_advantage"], row["rank"]))
    return output


def emit(name, rows, exact, order_ids, champion_roots, records_by_order):
    candidate_actions = []
    for index, row in enumerate(rows, 1):
        candidate_actions.append({
            "action_id": f"{name}_{index:03d}", "order_id": row["order_id"],
            "remove_rids": [row["rid"]] if row["kind"] == "delete" else [],
            "add_rids": [row["rid"]] if row["kind"] == "add" else [],
            "source": "v50_empirical_bayes_stratum", "expected_gain": row["estimated_advantage"],
            "evidence": row,
        })
    actions = list(exact["actions"]) + candidate_actions
    roots = apply_actions(champion_roots, records_by_order, actions)
    predictions = sum(map(len, roots.values()))
    path = OUT / "submissions" / f"{name}.csv"
    write_submission(path, order_ids, roots)
    possibilities = []
    if rows[0]["kind"] == "add":
        for correct in range(len(rows) + 1):
            tp = BASE_TP + correct
            score = 2 * tp / (TRUE_ROOTS + predictions)
            possibilities.append({
                "true_adds": correct, "batch_delta_tp": correct, "tp": tp,
                "score": round(score, 9),
                "improves_exact_base": score > 2 * BASE_TP / (TRUE_ROOTS + BASE_P),
            })
    else:
        for lost in range(len(rows) + 1):
            false_deletes = len(rows) - lost
            tp = BASE_TP - lost
            score = 2 * tp / (TRUE_ROOTS + predictions)
            possibilities.append({
                "deleted_true": lost, "false_deletes": false_deletes,
                "batch_delta_tp": -lost, "tp": tp, "score": round(score, 9),
                "improves_exact_base": score > 2 * BASE_TP / (TRUE_ROOTS + BASE_P),
            })
    manifest = {
        "probe_id": name, "baseline": str(CHAMPION), "fixed_base": exact["probe_id"],
        "path": str(path), "kind": rows[0]["kind"], "candidate_count": len(rows),
        "actions": actions, "predictions": predictions, "sha256": sha256(path),
        "estimated_correct": sum(row["matched_rule"]["posterior_mean"] for row in rows),
        "estimated_advantage": sum(row["estimated_advantage"] for row in rows),
        "score_possibilities": possibilities,
    }
    (OUT / "manifests" / f"{name}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "submissions").mkdir(exist_ok=True)
    (OUT / "manifests").mkdir(exist_ok=True)
    with np.load(DATA, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    with gzip.open(RECORDS, "rt", encoding="utf-8") as handle:
        records = json.load(handle)
    train_scores, test_scores = score_arrays(arrays, "train"), score_arrays(arrays, "test")
    train_average, test_average = rank_average(train_scores), rank_average(test_scores)
    train_base = exact_count_mask(train_average, arrays["train_alarm_ptr"], TRAIN_BUDGET)
    order_ids, champion_roots = load_submission(CHAMPION)
    test_base = selected_mask(records["test"], arrays["test_alarm_ptr"], champion_roots)
    labels = arrays["train_labels"].astype(np.int8)
    folds = arrays["train_station_folds"].astype(np.int8)
    exact = read_json(V37 / "manifests/v37_exact_corrections.json")
    v49 = read_json(V49 / "report.json")["manifest"]
    v49_candidate_actions = [action for action in v49["actions"] if action["source"] != "historical_leaderboard_equations"]
    excluded_orders = {action["order_id"] for action in exact["actions"] + v49_candidate_actions}
    scored = collect_scored_submissions()
    historical, _, _, _ = build_system(scored, root_nodes(champion_roots))
    excluded_nodes = set(historical) | protected_nodes()
    records_by_order = {row["order_id"]: row for row in records["test"]}
    ratio = BASE_TP / (TRUE_ROOTS + BASE_P)
    thresholds = {"add": ratio, "delete": 1 - ratio}

    kind_reports, pool = {}, []
    for kind in ("add", "delete"):
        train_rows, _ = unique_boundary(
            train_base, arrays["train_alarm_ptr"], train_average, kind, maximum=500
        )
        test_rows, _ = unique_boundary(
            test_base, arrays["test_alarm_ptr"], test_average, kind, maximum=500
        )
        train_table = candidate_table(
            train_rows, records["train"], arrays["train_alarm_ptr"], train_average,
            train_scores, train_base, labels, folds, kind,
        )
        test_table = candidate_table(
            test_rows, records["test"], arrays["test_alarm_ptr"], test_average,
            test_scores, test_base, kind=kind,
        )
        rules = exploratory_rules(train_table, thresholds[kind])
        candidates = select_candidates(test_table, rules, excluded_nodes, excluded_orders)
        kind_reports[kind] = {
            "threshold": thresholds[kind], "rule_count": len(rules),
            "rules": list(rules.values()), "candidate_count": len(candidates),
            "candidates": candidates,
        }
        pool.extend(candidates)

    # Assign each order to its highest expected-advantage action globally.
    pool.sort(key=lambda row: (-row["estimated_advantage"], row["rank"]))
    unique, used_orders = [], set(excluded_orders)
    for row in pool:
        if row["order_id"] in used_orders:
            continue
        unique.append(row)
        used_orders.add(row["order_id"])
    by_kind = {
        kind: [row for row in unique if row["kind"] == kind]
        for kind in ("add", "delete")
    }
    probes, batch_plan = {}, []
    for kind in ("add", "delete"):
        rows = by_kind[kind]
        for batch_index, start in enumerate(range(0, len(rows), BATCH_SIZE), 1):
            if batch_index > MAX_BATCHES_PER_KIND:
                break
            subset = rows[start:start + BATCH_SIZE]
            if len(subset) < 4:
                break
            name = f"v50_{kind}_batch{batch_index:02d}_n{len(subset):02d}_hold"
            manifest = emit(name, subset, exact, order_ids, champion_roots, records_by_order)
            probes[name] = manifest
            batch_plan.append({
                "probe_id": name, "kind": kind, "candidate_count": len(subset),
                "estimated_correct": manifest["estimated_correct"],
                "estimated_advantage": manifest["estimated_advantage"],
                "path": manifest["path"], "predictions": manifest["predictions"],
                "sha256": manifest["sha256"],
            })
    batch_plan.sort(key=lambda row: -row["estimated_advantage"])
    all_action_orders = [
        action["order_id"] for manifest in probes.values()
        for action in manifest["actions"] if action["source"] == "v50_empirical_bayes_stratum"
    ]
    if len(all_action_orders) != len(set(all_action_orders)):
        raise RuntimeError("V50 batches are not order-disjoint")
    v49_adds = sum(len(action["add_rids"]) for action in v49_candidate_actions)
    v49_deletes = sum(len(action["remove_rids"]) for action in v49_candidate_actions)
    v50_adds = sum(1 for row in unique if row["kind"] == "add")
    v50_deletes = sum(1 for row in unique if row["kind"] == "delete")
    optimistic_tp = BASE_TP + v49_adds + v50_adds
    optimistic_predictions = BASE_P + v49_adds + v50_adds - v49_deletes - v50_deletes
    optimistic_score = 2 * optimistic_tp / (TRUE_ROOTS + optimistic_predictions)
    report = {
        "version": "v50-disjoint-batch-campaign-1",
        "base": {"tp": BASE_TP, "predictions": BASE_P, "score": 2 * BASE_TP / (TRUE_ROOTS + BASE_P)},
        "exclusions": {
            "historical_nodes": len(historical), "historical_or_protected_nodes": len(excluded_nodes),
            "v37_v49_orders": len(excluded_orders),
        },
        "kinds": kind_reports,
        "unique_candidate_count": len(unique),
        "batch_plan": batch_plan,
        "probes": probes,
        "pool_upper_bound": {
            "assumption": "all candidate additions are true and all candidate deletions are false",
            "v49_additions": v49_adds, "v49_deletions": v49_deletes,
            "v50_additions": v50_adds, "v50_deletions": v50_deletes,
            "tp": optimistic_tp, "predictions": optimistic_predictions,
            "score": optimistic_score, "reaches_0_95": optimistic_score >= 0.95,
        },
        "warning": "These are quantitative online probes. Empirical-Bayes estimates rank batches but never authorize acceptance; accept only from reconstructed integer TP/F1.",
    }
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "rules": {kind: row["rule_count"] for kind, row in kind_reports.items()},
        "candidates": {kind: len(by_kind[kind]) for kind in by_kind},
        "batch_plan": batch_plan,
        "pool_upper_bound": report["pool_upper_bound"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
