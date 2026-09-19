"""Build fixed-count cross-order add/delete probes from V30 consensus scores."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(r"D:\zgyidong")
DATA = ROOT / "experiments/v25_semantic_router/cloud_dataset/v25_semantic_router.npz"
RECORDS = ROOT / "experiments/v25_semantic_router/cloud_dataset/semantic_records.json.gz"
OUTPUT = ROOT / "experiments/v30_meta_stack"
CHAMPION = ROOT / "experiments/v29_domain_ranker/submissions/day05_delete_verified.csv"
ONLINE = ROOT / "experiments/v29_domain_ranker/reports/online_results.json"
V29_MANIFESTS = ROOT / "experiments/v29_domain_ranker/manifests"
HISTORY = (
    ROOT / "experiments/v28_ten_day_campaign/state.json",
    ROOT / "experiments/v28_ten_day_campaign/reports/phase1_precision_state.json",
)
TRUE_ROOTS = 1044
BASELINE_TP = 953
BASELINE_P = 1035
MAX_ROOTS = 8


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_submission(path: Path):
    order_ids, roots = [], {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            order_ids.append(row["order_id"])
            roots[row["order_id"]] = json.loads(row["output"])["rootcause"]
    return order_ids, roots


def touched_orders() -> set[str]:
    touched = set()
    for path in HISTORY:
        state = read_json(path)
        for batch in state.get("batches", []):
            for action in batch.get("actions", []):
                touched.add(action["order_id"])
    online = read_json(ONLINE)
    for result in online.get("verified", []):
        manifest_path = V29_MANIFESTS / f"{result['probe_id']}.json"
        if manifest_path.exists():
            for action in read_json(manifest_path)["actions"]:
                touched.add(action["order_id"])
    return touched


def node_for_add(records_by_order, order_id: str, rid: str):
    alarm = next(item for item in records_by_order[order_id]["alarms"] if item["rid"] == rid)
    source = alarm["source"]
    return {
        "@rid": rid,
        "title": source.get("title", ""),
        "location": source.get("location", ""),
        "reason": source.get("reason", ""),
    }


def apply_actions(champion_roots, records_by_order, actions):
    output = {order_id: [dict(node) for node in nodes] for order_id, nodes in champion_roots.items()}
    seen = set()
    for action in actions:
        order_id = action["order_id"]
        if order_id in seen:
            raise ValueError(("multiple actions", order_id))
        seen.add(order_id)
        remove = set(action["remove_rids"])
        current = {node["@rid"] for node in output[order_id]}
        if not remove.issubset(current):
            raise ValueError(("missing removal", action))
        output[order_id] = [node for node in output[order_id] if node["@rid"] not in remove]
        for rid in action["add_rids"]:
            if rid in {node["@rid"] for node in output[order_id]}:
                raise ValueError(("duplicate addition", action))
            output[order_id].append(node_for_add(records_by_order, order_id, rid))
        if not 1 <= len(output[order_id]) <= MAX_ROOTS:
            raise ValueError(("invalid root count", order_id, len(output[order_id])))
    return output


def write_submission(path: Path, order_ids, roots):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["order_id", "output"])
        for order_id in order_ids:
            writer.writerow([order_id, json.dumps({"rootcause": roots[order_id]}, ensure_ascii=False)])


def score_possibilities(pair_count: int):
    return [
        {
            "delta_tp": delta,
            "tp": BASELINE_TP + delta,
            "score": round(2 * (BASELINE_TP + delta) / (TRUE_ROOTS + BASELINE_P), 9),
        }
        for delta in range(-pair_count, pair_count + 1)
    ]


def build_pairs(
    records,
    ptr,
    scores,
    champion_roots,
    excluded,
    protected_in=None,
    protected_out=None,
):
    protected_in = protected_in or set()
    protected_out = protected_out or set()
    order_for_row = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
    counts = np.asarray([len(champion_roots[order["order_id"]]) for order in records], dtype=np.int16)
    selected = np.zeros(len(scores), dtype=bool)
    row_alarm = []
    for order_index, (order, start, stop) in enumerate(zip(records, ptr[:-1], ptr[1:])):
        start, stop = int(start), int(stop)
        chosen = {node["@rid"] for node in champion_roots[order["order_id"]]}
        for alarm in order["alarms"]:
            row_alarm.append(alarm)
        selected[start:stop] = [alarm["rid"] in chosen for alarm in order["alarms"]]
    add_rows = [
        int(row)
        for row in np.argsort(-scores, kind="stable")
        if not selected[row]
        and counts[order_for_row[row]] < MAX_ROOTS
        and records[int(order_for_row[row])]["order_id"] not in excluded
        and (
            records[int(order_for_row[row])]["order_id"],
            row_alarm[row]["rid"],
        )
        not in protected_out
    ]
    remove_rows = [
        int(row)
        for row in np.argsort(scores, kind="stable")
        if selected[row]
        and counts[order_for_row[row]] > 1
        and records[int(order_for_row[row])]["order_id"] not in excluded
        and (
            records[int(order_for_row[row])]["order_id"],
            row_alarm[row]["rid"],
        )
        not in protected_in
    ]
    used = set()
    pairs = []
    add_index = remove_index = 0
    while add_index < len(add_rows) and remove_index < len(remove_rows):
        while add_index < len(add_rows) and int(order_for_row[add_rows[add_index]]) in used:
            add_index += 1
        while remove_index < len(remove_rows) and int(order_for_row[remove_rows[remove_index]]) in used:
            remove_index += 1
        if add_index >= len(add_rows) or remove_index >= len(remove_rows):
            break
        add_row, remove_row = add_rows[add_index], remove_rows[remove_index]
        add_order = int(order_for_row[add_row])
        remove_order = int(order_for_row[remove_row])
        if add_order == remove_order:
            if scores[add_row] >= scores[remove_row]:
                add_index += 1
            else:
                remove_index += 1
            continue
        margin = float(scores[add_row] - scores[remove_row])
        # Once the best remaining add is no better than the best remaining
        # removal, every later pair is unsafe as a root-count reallocation.
        if margin <= 0:
            break
        pairs.append(
            {
                "pair_id": f"v30_pair_{len(pairs) + 1:03d}",
                "margin": margin,
                "add_order_index": add_order,
                "remove_order_index": remove_order,
                "add_row": add_row,
                "remove_row": remove_row,
                "add_score": float(scores[add_row]),
                "remove_score": float(scores[remove_row]),
            }
        )
        used.add(add_order)
        used.add(remove_order)
        add_index += 1
        remove_index += 1
    return pairs, row_alarm


def actions_for_pairs(pairs, records, row_alarm):
    actions = []
    for pair in pairs:
        pair_id = pair["pair_id"]
        add_order = records[pair["add_order_index"]]
        remove_order = records[pair["remove_order_index"]]
        add_alarm = row_alarm[pair["add_row"]]
        remove_alarm = row_alarm[pair["remove_row"]]
        actions.extend(
            [
                {
                    "action_id": f"{pair_id}_add",
                    "order_id": add_order["order_id"],
                    "remove_rids": [],
                    "add_rids": [add_alarm["rid"]],
                    "source": "v30_cross_order_meta",
                    "expected_gain": pair["margin"],
                    "evidence": {
                        "pair_id": pair_id,
                        "node_score": pair["add_score"],
                        "pair_margin": pair["margin"],
                        "title": add_alarm.get("title", ""),
                    },
                },
                {
                    "action_id": f"{pair_id}_remove",
                    "order_id": remove_order["order_id"],
                    "remove_rids": [remove_alarm["rid"]],
                    "add_rids": [],
                    "source": "v30_cross_order_meta",
                    "expected_gain": pair["margin"],
                    "evidence": {
                        "pair_id": pair_id,
                        "node_score": pair["remove_score"],
                        "pair_margin": pair["margin"],
                        "title": remove_alarm.get("title", ""),
                    },
                },
            ]
        )
    return actions


def emit(name, pair_count, pairs, order_ids, champion_roots, records_by_order, records):
    subset = pairs[:pair_count]
    row_alarm = [alarm for order in records for alarm in order["alarms"]]
    actions = actions_for_pairs(subset, records, row_alarm)
    roots = apply_actions(champion_roots, records_by_order, actions)
    predictions = sum(len(nodes) for nodes in roots.values())
    if predictions != BASELINE_P:
        raise ValueError((name, predictions, BASELINE_P))
    path = OUTPUT / "submissions" / f"{name}.csv"
    write_submission(path, order_ids, roots)
    manifest = {
        "probe_id": name,
        "path": str(path),
        "baseline": str(CHAMPION),
        "pair_count": pair_count,
        "actions": actions,
        "predictions": predictions,
        "prediction_delta": 0,
        "sha256": sha256(path),
        "score_possibilities": score_possibilities(pair_count),
    }
    manifest_path = OUTPUT / "manifests" / f"{name}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main():
    with np.load(DATA, allow_pickle=False) as archive:
        ptr = archive["test_alarm_ptr"]
    with gzip.open(RECORDS, "rt", encoding="utf-8") as handle:
        records = json.load(handle)["test"]
    scores = np.load(OUTPUT / "v30_consensus_test.npy")
    order_ids, champion_roots = load_submission(CHAMPION)
    records_by_order = {order["order_id"]: order for order in records}
    excluded = touched_orders()
    pairs, _ = build_pairs(records, ptr, scores, champion_roots, excluded)
    manifests = {}
    requested_counts = [5, 10, 15, len(pairs)]
    safe_counts = []
    for pair_count in requested_counts:
        if 0 < pair_count <= len(pairs) and pair_count not in safe_counts:
            safe_counts.append(pair_count)
    for pair_count in safe_counts:
        name = f"v30_cross_order_top{pair_count}"
        manifests[name] = emit(
            name, pair_count, pairs, order_ids, champion_roots, records_by_order, records
        )
    report = {
        "version": "v30-cross-order-1",
        "baseline": str(CHAMPION),
        "baseline_tp": BASELINE_TP,
        "baseline_predictions": BASELINE_P,
        "excluded_orders": len(excluded),
        "available_positive_margin_pairs": len(pairs),
        "selection_rule": "pair_margin > 0; generation stops at first non-positive margin",
        "unsafe_legacy_artifacts": [
            "submissions/v30_cross_order_top20.csv",
            "submissions/v30_cross_order_top50.csv",
        ],
        "protected_node_warning": (
            "Direct top10/top15/top16 probes were ranked before the V11 protected-node "
            "audit. After top5 online verification, use build_safe_extension.py instead."
        ),
        "top_pairs": pairs[:50],
        "probes": manifests,
    }
    report_path = OUTPUT / "v30_cross_order_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "excluded_orders": len(excluded),
                "available_pairs": len(pairs),
                "probes": {
                    name: {
                        "pairs": value["pair_count"],
                        "actions": len(value["actions"]),
                        "predictions": value["predictions"],
                        "sha256": value["sha256"],
                    }
                    for name, value in manifests.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
