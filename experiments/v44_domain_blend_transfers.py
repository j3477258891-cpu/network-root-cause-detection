"""Extract conservative cross-order transfers from the V43 domain blend."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / ".deps"
V30 = ROOT / "experiments/v30_meta_stack"
EXPERIMENTS = ROOT / "experiments"
for value in (DEPS, V30, EXPERIMENTS):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from build_cross_order_probes import apply_actions, load_submission, write_submission
from v37_online_equation_solver import build_system, collect_scored_submissions
from v43_low_dim_calibration import exact_budget_mask


DATA = EXPERIMENTS / "v25_semantic_router/cloud_dataset/v25_semantic_router.npz"
RECORDS = EXPERIMENTS / "v25_semantic_router/cloud_dataset/semantic_records.json.gz"
CHAMPION = V30 / "submissions/v30_cross_order_top5.csv"
V37 = EXPERIMENTS / "v37_online_equations"
V11_REPORT = EXPERIMENTS / "submissions/template_ranked/v11_constrained_report.json"
OUT = EXPERIMENTS / "v44_domain_blend_transfers"
TRUE_ROOTS = 1044
BASE_TP_AFTER_EXACT = 956
BASE_P = 1035
MAX_ROOTS = 8
STATION_WEIGHT = 0.4


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def root_nodes(roots):
    return {(oid, node["@rid"]) for oid, values in roots.items() for node in values}


def protected_nodes():
    report = read_json(V11_REPORT)
    output = set()
    for key in ("protected_in", "protected_out"):
        output |= {(row["order_id"], row["rid"]) for row in report.get(key, [])}
    return output


def greedy_pairs(score, selected, ptr, excluded_rows=None, excluded_orders=None):
    excluded_rows = excluded_rows or set()
    excluded_orders = excluded_orders or set()
    order_for_row = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
    counts = np.asarray([
        int(selected[int(start):int(stop)].sum())
        for start, stop in zip(ptr[:-1], ptr[1:])
    ], dtype=np.int16)
    add_rows = [
        int(row) for row in np.argsort(-score, kind="stable")
        if not selected[row]
        and counts[order_for_row[row]] < MAX_ROOTS
        and int(row) not in excluded_rows
        and int(order_for_row[row]) not in excluded_orders
    ]
    remove_rows = [
        int(row) for row in np.argsort(score, kind="stable")
        if selected[row]
        and counts[order_for_row[row]] > 1
        and int(row) not in excluded_rows
        and int(order_for_row[row]) not in excluded_orders
    ]
    used, pairs = set(), []
    ai = ri = 0
    while ai < len(add_rows) and ri < len(remove_rows):
        while ai < len(add_rows) and int(order_for_row[add_rows[ai]]) in used:
            ai += 1
        while ri < len(remove_rows) and int(order_for_row[remove_rows[ri]]) in used:
            ri += 1
        if ai >= len(add_rows) or ri >= len(remove_rows):
            break
        add_row, remove_row = add_rows[ai], remove_rows[ri]
        add_order = int(order_for_row[add_row])
        remove_order = int(order_for_row[remove_row])
        if add_order == remove_order:
            if score[add_row] >= score[remove_row]:
                ai += 1
            else:
                ri += 1
            continue
        margin = float(score[add_row] - score[remove_row])
        if margin <= 0:
            break
        pairs.append({
            "pair_index": len(pairs) + 1,
            "add_row": add_row,
            "remove_row": remove_row,
            "add_order_index": add_order,
            "remove_order_index": remove_order,
            "add_score": float(score[add_row]),
            "remove_score": float(score[remove_row]),
            "margin": margin,
        })
        used.add(add_order)
        used.add(remove_order)
        ai += 1
        ri += 1
    return pairs, order_for_row


def oof_curve(pairs, labels):
    output = {}
    for k in (1, 2, 3, 5, 8, 10, 16, 24, 32):
        local = pairs[:min(k, len(pairs))]
        if not local:
            continue
        deltas = [int(labels[row["add_row"]]) - int(labels[row["remove_row"]]) for row in local]
        output[str(len(local))] = {
            "delta_tp": int(sum(deltas)),
            "positive_pairs": int(sum(value > 0 for value in deltas)),
            "neutral_pairs": int(sum(value == 0 for value in deltas)),
            "negative_pairs": int(sum(value < 0 for value in deltas)),
            "deltas": deltas,
            "minimum_margin": min(row["margin"] for row in local),
        }
    return output


def test_exclusions(records, ptr, historical, protected, exact_orders):
    excluded_rows, excluded_order_indices = set(), set()
    row = 0
    for oi, order in enumerate(records):
        oid = order["order_id"]
        if oid in exact_orders:
            excluded_order_indices.add(oi)
        for alarm in order["alarms"]:
            if (oid, alarm["rid"]) in historical or (oid, alarm["rid"]) in protected:
                excluded_rows.add(row)
            row += 1
    if row != int(ptr[-1]):
        raise ValueError((row, int(ptr[-1])))
    return excluded_rows, excluded_order_indices


def pair_actions(pairs, records):
    actions = []
    for index, pair in enumerate(pairs, 1):
        add_order = records[pair["add_order_index"]]
        remove_order = records[pair["remove_order_index"]]
        add_start = sum(len(row["alarms"]) for row in records[:pair["add_order_index"]])
        remove_start = sum(len(row["alarms"]) for row in records[:pair["remove_order_index"]])
        add_alarm = add_order["alarms"][pair["add_row"] - add_start]
        remove_alarm = remove_order["alarms"][pair["remove_row"] - remove_start]
        pair_id = f"v44_pair_{index:03d}"
        evidence = {
            "pair_id": pair_id, "margin": pair["margin"],
            "add_score": pair["add_score"], "remove_score": pair["remove_score"],
        }
        actions.extend([
            {
                "action_id": f"{pair_id}_add", "order_id": add_order["order_id"],
                "remove_rids": [], "add_rids": [add_alarm["rid"]],
                "source": "v44_station_domain_blend", "expected_gain": pair["margin"],
                "evidence": {**evidence, "title": add_alarm.get("title", "")},
            },
            {
                "action_id": f"{pair_id}_remove", "order_id": remove_order["order_id"],
                "remove_rids": [remove_alarm["rid"]], "add_rids": [],
                "source": "v44_station_domain_blend", "expected_gain": pair["margin"],
                "evidence": {**evidence, "title": remove_alarm.get("title", "")},
            },
        ])
    return actions


def emit(name, pairs, exact, order_ids, champion_roots, records):
    actions = list(exact["actions"]) + pair_actions(pairs, records)
    records_by_order = {row["order_id"]: row for row in records}
    roots = apply_actions(champion_roots, records_by_order, actions)
    predictions = sum(map(len, roots.values()))
    if predictions != BASE_P:
        raise ValueError((name, predictions))
    path = OUT / "submissions" / f"{name}.csv"
    write_submission(path, order_ids, roots)
    pair_count = len(pairs)
    manifest = {
        "probe_id": name, "baseline": str(CHAMPION), "fixed_base": exact["probe_id"],
        "path": str(path), "pair_count": pair_count, "actions": actions,
        "predictions": predictions, "sha256": sha256(path),
        "score_possibilities": [
            {
                "candidate_delta_tp": delta,
                "tp": BASE_TP_AFTER_EXACT + delta,
                "score": round(2 * (BASE_TP_AFTER_EXACT + delta) / (TRUE_ROOTS + predictions), 9),
            }
            for delta in range(-pair_count, pair_count + 1)
        ],
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
    train_ptr, test_ptr = arrays["train_alarm_ptr"], arrays["test_alarm_ptr"]
    labels = arrays["train_labels"].astype(bool)
    target_count = np.asarray([
        min(MAX_ROOTS, int(labels[int(start):int(stop)].sum()))
        for start, stop in zip(train_ptr[:-1], train_ptr[1:])
    ], dtype=np.int16)
    train_budget = int(target_count.sum())
    station_oof = np.load(V30 / "station_extra_trees_oof.npy")
    station_test = np.load(V30 / "station_extra_trees_test.npy")
    domain_oof = np.load(EXPERIMENTS / "v33_domain_adaptation/station_alpha_2_oof.npy")
    domain_test = np.load(EXPERIMENTS / "v33_domain_adaptation/station_alpha_2_test.npy")
    blend_oof = STATION_WEIGHT * station_oof + (1 - STATION_WEIGHT) * domain_oof
    blend_test = STATION_WEIGHT * station_test + (1 - STATION_WEIGHT) * domain_test

    train_base = exact_budget_mask(station_oof, train_ptr, train_budget)
    train_pairs, _ = greedy_pairs(blend_oof, train_base, train_ptr)
    curve = oof_curve(train_pairs, labels)
    gate_passed = bool(
        curve.get("5", {}).get("delta_tp", -99) >= 3
        and curve.get("5", {}).get("negative_pairs", 99) <= 1
        and curve.get("10", {}).get("delta_tp", -99) >= 4
    )

    with gzip.open(RECORDS, "rt", encoding="utf-8") as handle:
        records = json.load(handle)["test"]
    order_ids, champion_roots = load_submission(CHAMPION)
    champion_selected = np.zeros(len(blend_test), dtype=bool)
    row = 0
    for order in records:
        chosen = {node["@rid"] for node in champion_roots[order["order_id"]]}
        for alarm in order["alarms"]:
            champion_selected[row] = alarm["rid"] in chosen
            row += 1
    if row != len(blend_test):
        raise ValueError((row, len(blend_test)))

    scored = collect_scored_submissions()
    keys, _, _, _ = build_system(scored, root_nodes(champion_roots))
    historical = set(keys)
    protected = protected_nodes()
    exact = read_json(V37 / "manifests/v37_exact_corrections.json")
    exact_orders = {action["order_id"] for action in exact["actions"]}
    excluded_rows, excluded_orders = test_exclusions(
        records, test_ptr, historical, protected, exact_orders
    )
    test_pairs, _ = greedy_pairs(
        blend_test, champion_selected, test_ptr, excluded_rows, excluded_orders
    )
    probes = {}
    if gate_passed:
        for count in (3, 5, 8):
            if len(test_pairs) >= count:
                name = f"v44_domain_transfer_top{count}_hold"
                probes[name] = emit(
                    name, test_pairs[:count], exact, order_ids, champion_roots, records
                )
    report = {
        "version": "v44-domain-blend-transfers-1",
        "blend": {"station_weight": STATION_WEIGHT, "domain_alpha2_weight": 1 - STATION_WEIGHT},
        "oof_available_pairs": len(train_pairs),
        "oof_curve": curve,
        "exclusions": {
            "historical_nodes": len(historical), "protected_nodes": len(protected),
            "excluded_test_rows": len(excluded_rows), "exact_orders": len(exact_orders),
        },
        "test_available_pairs": len(test_pairs),
        "test_top_pairs": test_pairs[:30],
        "gate": {
            "requires_top5_delta_tp": 3,
            "requires_top5_negative_pairs_at_most": 1,
            "requires_top10_delta_tp": 4,
            "passed": gate_passed,
        },
        "probes": probes,
    }
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "oof_curve": curve,
        "gate": report["gate"],
        "test_available_pairs": len(test_pairs),
        "probes": {name: {
            "path": row["path"], "pairs": row["pair_count"],
            "sha256": row["sha256"],
        } for name, row in probes.items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
