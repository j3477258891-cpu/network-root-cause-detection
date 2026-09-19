"""Build disjoint, small-count frontier probes from unused boundary actions.

The probes are anchored to the V37 exact correction and never reuse an order
from V47/V49/V50.  Each probe has four actions of one kind so its leaderboard
score identifies the batch count without mixing add and delete semantics.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
V30 = EXPERIMENTS / "v30_meta_stack"
if str(V30) not in sys.path:
    sys.path.insert(0, str(V30))

from build_cross_order_probes import apply_actions, load_submission, write_submission
from v37_online_equation_solver import build_system, collect_scored_submissions


CHAMPION = V30 / "submissions/v30_cross_order_top5.csv"
RECORDS = EXPERIMENTS / "v25_semantic_router/cloud_dataset/semantic_records.json.gz"
V37 = EXPERIMENTS / "v37_online_equations"
V36 = EXPERIMENTS / "v36_selective_counts/report.json"
V40 = EXPERIMENTS / "v40_one_sided_boundary/report.json"
V11 = EXPERIMENTS / "submissions/template_ranked/v11_constrained_report.json"
OUT = EXPERIMENTS / "v52_precision_frontier"
TRUE_ROOTS = 1044
BASE_TP = 956
BASE_P = 1035
MAX_ROOTS = 8
BATCH_SIZE = 4
MAX_BATCHES_PER_KIND = 8


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def action_nodes(action: dict):
    oid = action["order_id"]
    return {(oid, rid) for rid in action.get("remove_rids", []) + action.get("add_rids", [])}


def existing_actions():
    actions = []
    for directory in (
        EXPERIMENTS / "v37_online_equations/manifests",
        EXPERIMENTS / "v47_v36_count_batch/manifests",
        EXPERIMENTS / "v49_extended_count_batch/manifests",
        EXPERIMENTS / "v50_disjoint_batches/manifests",
    ):
        for path in sorted(directory.glob("*.json")):
            data = read_json(path)
            actions.extend(data.get("actions", []))
    return actions


def historical_variables():
    order_ids, roots = load_submission(CHAMPION)
    champion_nodes = {(oid, node["@rid"]) for oid, values in roots.items() for node in values}
    keys, _, _, _ = build_system(collect_scored_submissions(), champion_nodes)
    return set(keys)


def protected_variables():
    output = set()
    if not V11.exists():
        return output
    report = read_json(V11)
    for name in ("protected_in", "protected_out", "forced_in", "forced_out"):
        output |= {(row["order_id"], row["rid"]) for row in report.get(name, [])}
    return output


def make_action(row: dict, source: str, rank: int, quality: float):
    kind = row["kind"]
    action = {
        "action_id": f"v52_{kind}_{rank:03d}_{row['order_id'][:8]}",
        "order_id": row["order_id"],
        "remove_rids": [row["rid"]] if kind == "delete" else [],
        "add_rids": [row["rid"]] if kind == "add" else [],
        "source": source,
        "expected_gain": float(quality),
        "evidence": {"source_rank": rank, **row},
    }
    return action


def collect_pool(existing_orders, forbidden_nodes):
    v40 = read_json(V40)
    v36 = read_json(V36)
    candidates = {"add": [], "delete": []}

    # Preserve V40's validated frontier order: its first ten additions have
    # the strongest nested OOF precision in the existing audits.
    for kind in ("add", "delete"):
        for rank, row in enumerate(v40["test_candidates"][kind], 1):
            key = (row["order_id"], row["rid"])
            if row["order_id"] in existing_orders or key in forbidden_nodes:
                continue
            candidates[kind].append(
                (3000.0 - rank, make_action(row, "v40_boundary_frontier", rank, 1.0 / rank))
            )

    # Count-model actions are independent evidence.  Add only rows not already
    # represented by the V40 frontier; higher margin gets earlier placement.
    for row in v36["accepted_candidates"] + v36["rejected_candidates"]:
        kind = row["kind"]
        key = (row["order_id"], row["rid"])
        if row["order_id"] in existing_orders or key in forbidden_nodes:
            continue
        candidates[kind].append(
            (2000.0 + float(row.get("mean_margin", 0.0)),
             make_action(row, "v36_count_frontier", 0, float(row.get("mean_margin", 0.0))))
        )

    output = {"add": [], "delete": []}
    used_nodes = set(forbidden_nodes)
    used_orders = set(existing_orders)
    # Add frontier actions first, then fill with count actions.  This also
    # guarantees one action per order across both directions.
    for kind in ("add", "delete"):
        candidates[kind].sort(key=lambda item: (-item[0], item[1]["order_id"], item[1]["action_id"]))
    merged = sorted(
        [(priority, kind, action) for kind, rows in candidates.items() for priority, action in rows],
        key=lambda item: (-item[0], 0 if item[1] == "add" else 1, item[2]["order_id"], item[2]["action_id"]),
    )
    for _, kind, action in merged:
        nodes = action_nodes(action)
        if action["order_id"] in used_orders or nodes & used_nodes:
            continue
        used_orders.add(action["order_id"])
        used_nodes |= nodes
        output[kind].append(action)
    return output


def score_table(kind: str, count: int):
    rows = []
    for correct in range(count + 1):
        tp = BASE_TP + correct if kind == "add" else BASE_TP - correct
        rows.append({
            "candidate_correct": correct,
            "tp": tp,
            "predictions_delta": count if kind == "add" else -count,
            "score": round(2 * tp / (TRUE_ROOTS + BASE_P + (count if kind == "add" else -count)), 9),
        })
    return rows


def emit_probe(kind, index, actions, order_ids, champion_roots, records_by_order, exact):
    name = f"v52_{kind}_batch{index:02d}_n{len(actions):02d}"
    all_actions = [dict(action) for action in exact["actions"]] + [dict(action) for action in actions]
    roots = apply_actions(champion_roots, records_by_order, all_actions)
    predictions = sum(len(values) for values in roots.values())
    expected_predictions = BASE_P + (len(actions) if kind == "add" else -len(actions))
    if predictions != expected_predictions:
        raise RuntimeError((name, predictions, expected_predictions))
    path = OUT / "submissions" / f"{name}.csv"
    write_submission(path, order_ids, roots)
    manifest = {
        "probe_id": name,
        "baseline": str(CHAMPION),
        "fixed_base": exact["probe_id"],
        "path": str(path),
        "kind": kind,
        "candidate_count": len(actions),
        "actions": all_actions,
        "predictions": predictions,
        "sha256": sha256(path),
        "score_possibilities": score_table(kind, len(actions)),
    }
    (OUT / "manifests" / f"{name}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "submissions").mkdir(exist_ok=True)
    (OUT / "manifests").mkdir(exist_ok=True)
    order_ids, champion_roots = load_submission(CHAMPION)
    with gzip.open(RECORDS, "rt", encoding="utf-8") as handle:
        records = json.load(handle)["test"]
    records_by_order = {row["order_id"]: row for row in records}
    exact = read_json(V37 / "manifests/v37_exact_corrections.json")
    existing = existing_actions()
    existing_orders = {action["order_id"] for action in existing}
    forbidden = historical_variables() | protected_variables()
    for action in existing:
        forbidden |= action_nodes(action)
    pool = collect_pool(existing_orders, forbidden)
    # Keep the first 32 candidates per direction.  Eight four-action probes
    # fit the available daily budget while leaving room for V49/V47 checks.
    probes = {}
    for kind in ("add", "delete"):
        rows = pool[kind][: BATCH_SIZE * MAX_BATCHES_PER_KIND]
        for start in range(0, len(rows), BATCH_SIZE):
            batch = rows[start : start + BATCH_SIZE]
            if len(batch) < BATCH_SIZE:
                break
            index = start // BATCH_SIZE + 1
            probes[f"v52_{kind}_batch{index:02d}_n{BATCH_SIZE:02d}"] = emit_probe(
                kind, index, batch, order_ids, champion_roots, records_by_order, exact
            )
    report = {
        "version": "v52-precision-frontier-1",
        "base": {"tp": BASE_TP, "predictions": BASE_P, "score": 2 * BASE_TP / (TRUE_ROOTS + BASE_P)},
        "exclusions": {
            "existing_orders": len(existing_orders),
            "historical_or_protected_nodes": len(forbidden),
        },
        "pool_counts": {kind: len(rows) for kind, rows in pool.items()},
        "pool": pool,
        "probes": probes,
        "optimistic_all_batches": {
            "add_count": sum(m["candidate_count"] for n, m in probes.items() if m["kind"] == "add"),
            "delete_count": sum(m["candidate_count"] for n, m in probes.items() if m["kind"] == "delete"),
        },
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "pool_counts": report["pool_counts"],
        "probe_count": len(probes),
        "probes": {name: {k: row[k] for k in ("kind", "candidate_count", "predictions", "sha256")} for name, row in probes.items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
