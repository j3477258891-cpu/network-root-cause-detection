"""Learn station-stable boundary-candidate strata with nested validation."""

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


DATA = EXPERIMENTS / "v25_semantic_router/cloud_dataset/v25_semantic_router.npz"
RECORDS = EXPERIMENTS / "v25_semantic_router/cloud_dataset/semantic_records.json.gz"
CHAMPION = V30 / "submissions/v30_cross_order_top5.csv"
V37 = EXPERIMENTS / "v37_online_equations"
V11_REPORT = EXPERIMENTS / "submissions/template_ranked/v11_constrained_report.json"
OUT = EXPERIMENTS / "v48_stable_strata"
TRAIN_BUDGET = 3103
TRUE_ROOTS = 1044
BASE_TP_AFTER_EXACT = 956
BASE_P = 1035
MAX_ROOTS = 8
MIN_SUPPORT = 6
MIN_FOLDS = 3
BETA_QUANTILE = 0.10


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def root_nodes(roots):
    return {(oid, node["@rid"]) for oid, values in roots.items() for node in values}


def protected_nodes():
    report = read_json(V11_REPORT)
    output = set()
    for name in ("protected_in", "protected_out"):
        output |= {(row["order_id"], row["rid"]) for row in report.get(name, [])}
    return output


def rank_bin(rank):
    for limit, name in ((10, "r01_10"), (25, "r11_25"), (50, "r26_50"),
                        (100, "r51_100"), (200, "r101_200")):
        if rank <= limit:
            return name
    return "r201_plus"


def candidate_table(rows, records, ptr, average, scores, base, labels=None, folds=None, kind="add"):
    order_for_row = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
    local_for_row = np.concatenate([np.arange(int(stop - start)) for start, stop in zip(ptr[:-1], ptr[1:])])
    counts = np.asarray([base[int(start):int(stop)].sum() for start, stop in zip(ptr[:-1], ptr[1:])])
    output = []
    for rank, row in enumerate(rows, 1):
        oi = int(order_for_row[row])
        order = records[oi]
        alarm = order["alarms"][int(local_for_row[row])]
        score_values = {name: float(value[row]) for name, value in scores.items()}
        score_median = float(np.median(list(score_values.values())))
        agreement = int(sum(value >= 0.5 for value in score_values.values()))
        item = {
            "row": int(row), "rank": rank, "rank_bin": rank_bin(rank),
            "order_index": oi, "order_id": order["order_id"], "rid": alarm["rid"],
            "kind": kind, "title": alarm.get("title", ""),
            "vendor": alarm.get("vendor", ""), "device_type": alarm.get("device_type", ""),
            "deployment": alarm.get("deployment", ""), "radio": alarm.get("radio", ""),
            "timeline": alarm.get("timeline", ""), "current_count": int(counts[oi]),
            "alarm_count": len(order["alarms"]), "rank_average": float(average[row]),
            "score_median": score_median, "agreement": agreement,
            "scores": score_values,
        }
        if labels is not None:
            label = int(labels[row])
            item["target"] = label if kind == "add" else 1 - label
        if folds is not None:
            item["fold"] = int(folds[oi])
        output.append(item)
    return output


RULE_FIELDS = (
    ("rank_bin",),
    ("title",),
    ("device_type",),
    ("vendor",),
    ("deployment",),
    ("title", "device_type"),
    ("title", "vendor"),
    ("title", "deployment"),
    ("title", "rank_bin"),
    ("device_type", "rank_bin"),
    ("vendor", "rank_bin"),
    ("deployment", "rank_bin"),
    ("title", "device_type", "rank_bin"),
)


def rule_key(row, fields):
    values = tuple(str(row.get(field, "")) for field in fields)
    if any(not value for value in values):
        return None
    return fields, values


def learn_rules(rows, acceptance_threshold, min_support=MIN_SUPPORT, min_folds=MIN_FOLDS):
    grouped = defaultdict(list)
    for row in rows:
        for fields in RULE_FIELDS:
            key = rule_key(row, fields)
            if key is not None:
                grouped[key].append(row)
    output = {}
    for key, members in grouped.items():
        support = len(members)
        folds = sorted({row["fold"] for row in members})
        if support < min_support or len(folds) < min_folds:
            continue
        correct = sum(row["target"] for row in members)
        precision = correct / support
        fold_precision = {
            str(fold): float(np.mean([row["target"] for row in members if row["fold"] == fold]))
            for fold in folds
        }
        lower = float(beta_distribution.ppf(BETA_QUANTILE, 1 + correct, 1 + support - correct))
        if precision < acceptance_threshold + 0.08:
            continue
        if min(fold_precision.values()) < acceptance_threshold:
            continue
        if lower < acceptance_threshold:
            continue
        output[key] = {
            "fields": list(key[0]), "values": list(key[1]),
            "support": support, "correct": correct, "precision": precision,
            "fold_count": len(folds), "fold_precision": fold_precision,
            "beta_lower_90": lower,
        }
    return output


def match_best(row, rules):
    matches = []
    for fields in RULE_FIELDS:
        key = rule_key(row, fields)
        if key in rules:
            matches.append(rules[key])
    if not matches:
        return None
    return max(matches, key=lambda rule: (
        rule["beta_lower_90"], len(rule["fields"]), rule["support"]
    ))


def nested_audit(rows, acceptance_threshold):
    selected = []
    fold_reports = []
    for fold in range(5):
        train = [row for row in rows if row["fold"] != fold]
        valid = [row for row in rows if row["fold"] == fold]
        # Four-fold training sees roughly 80% of a full-data stratum. Scaling
        # support here prevents mechanically deleting every support-6 rule.
        rules = learn_rules(train, acceptance_threshold, min_support=4, min_folds=2)
        chosen = []
        for row in valid:
            rule = match_best(row, rules)
            if rule is not None:
                chosen.append({**row, "matched_rule": rule})
        selected.extend(chosen)
        fold_reports.append({
            "fold": fold, "learned_rules": len(rules), "selected": len(chosen),
            "correct": sum(row["target"] for row in chosen),
            "precision": float(np.mean([row["target"] for row in chosen])) if chosen else None,
        })
    represented = [row for row in fold_reports if row["selected"]]
    return {
        "selected": len(selected), "correct": sum(row["target"] for row in selected),
        "precision": float(np.mean([row["target"] for row in selected])) if selected else None,
        "minimum_fold_precision": min(row["precision"] for row in represented) if represented else None,
        "represented_folds": len(represented), "folds": fold_reports,
    }


def clean_test_matches(rows, rules, excluded_nodes, excluded_orders):
    output = []
    used_orders = set()
    for row in rows:
        key = (row["order_id"], row["rid"])
        if key in excluded_nodes or row["order_id"] in excluded_orders or row["order_id"] in used_orders:
            continue
        rule = match_best(row, rules)
        if rule is None:
            continue
        output.append({**row, "matched_rule": rule})
        used_orders.add(row["order_id"])
    output.sort(key=lambda row: (
        -row["matched_rule"]["beta_lower_90"],
        -row["matched_rule"]["precision"], row["rank"],
    ))
    return output


def emit(name, rows, exact, order_ids, champion_roots, records_by_order):
    candidate_actions = []
    for index, row in enumerate(rows, 1):
        candidate_actions.append({
            "action_id": f"{name}_{index:03d}", "order_id": row["order_id"],
            "remove_rids": [row["rid"]] if row["kind"] == "delete" else [],
            "add_rids": [row["rid"]] if row["kind"] == "add" else [],
            "source": "v48_station_stable_stratum", "expected_gain": None,
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
            tp = BASE_TP_AFTER_EXACT + correct
            possibilities.append({
                "true_adds": correct, "tp": tp,
                "score": round(2 * tp / (TRUE_ROOTS + predictions), 9),
            })
    else:
        for lost in range(len(rows) + 1):
            tp = BASE_TP_AFTER_EXACT - lost
            possibilities.append({
                "deleted_true": lost, "tp": tp,
                "score": round(2 * tp / (TRUE_ROOTS + predictions), 9),
            })
    manifest = {
        "probe_id": name, "baseline": str(CHAMPION), "fixed_base": exact["probe_id"],
        "path": str(path), "candidate_count": len(rows), "actions": actions,
        "predictions": predictions, "sha256": sha256(path),
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
    train_scores = score_arrays(arrays, "train")
    test_scores = score_arrays(arrays, "test")
    train_average = rank_average(train_scores)
    test_average = rank_average(test_scores)
    train_base = exact_count_mask(train_average, arrays["train_alarm_ptr"], TRAIN_BUDGET)
    order_ids, champion_roots = load_submission(CHAMPION)
    test_base = selected_mask(records["test"], arrays["test_alarm_ptr"], champion_roots)
    labels = arrays["train_labels"].astype(np.int8)
    folds = arrays["train_station_folds"].astype(np.int8)
    exact = read_json(V37 / "manifests/v37_exact_corrections.json")
    excluded_orders = {action["order_id"] for action in exact["actions"]}
    scored = collect_scored_submissions()
    historical_keys, _, _, _ = build_system(scored, root_nodes(champion_roots))
    excluded_nodes = set(historical_keys) | protected_nodes()
    records_by_order = {row["order_id"]: row for row in records["test"]}

    acceptance = {
        "add": BASE_TP_AFTER_EXACT / (TRUE_ROOTS + BASE_P),
        "delete": 1 - BASE_TP_AFTER_EXACT / (TRUE_ROOTS + BASE_P),
    }
    reports, probes = {}, {}
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
        nested = nested_audit(train_table, acceptance[kind])
        rules = learn_rules(train_table, acceptance[kind])
        matched = clean_test_matches(test_table, rules, excluded_nodes, excluded_orders)
        gate_passed = bool(
            nested["selected"] >= 6
            and nested["represented_folds"] >= 3
            and nested["precision"] is not None
            and nested["precision"] >= acceptance[kind] + 0.08
            and nested["minimum_fold_precision"] >= acceptance[kind]
            and len(matched) >= 4
        )
        reports[kind] = {
            "acceptance_threshold": acceptance[kind],
            "train_candidate_count": len(train_table), "nested_audit": nested,
            "full_rule_count": len(rules), "full_rules": list(rules.values()),
            "test_candidate_count": len(test_table), "clean_test_matches": matched,
            "gate": {
                "requires_nested_selected": 6,
                "requires_represented_folds": 3,
                "requires_precision_margin": 0.08,
                "requires_minimum_fold_at_threshold": True,
                "passed": gate_passed,
            },
        }
        if gate_passed:
            for count in (4, 8, 12, 16):
                if len(matched) < count:
                    continue
                name = f"v48_{kind}_stable_top{count:02d}_hold"
                probes[name] = emit(
                    name, matched[:count], exact, order_ids, champion_roots, records_by_order
                )
    report = {
        "version": "v48-stable-strata-1",
        "exclusions": {
            "historical_nodes": len(historical_keys),
            "historical_or_protected_nodes": len(excluded_nodes),
            "exact_orders": len(excluded_orders),
        },
        "kinds": reports, "probes": probes,
    }
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "kinds": {kind: {
            "nested_audit": row["nested_audit"],
            "full_rule_count": row["full_rule_count"],
            "clean_test_matches": len(row["clean_test_matches"]),
            "gate": row["gate"],
        } for kind, row in reports.items()},
        "probes": {name: {
            "path": row["path"], "candidate_count": row["candidate_count"],
            "predictions": row["predictions"], "sha256": row["sha256"],
        } for name, row in probes.items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
