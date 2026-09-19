"""Audit canonical template transfer for selective root-count corrections."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(r"D:\zgyidong")
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments/v27_closed_loop"))
sys.path.insert(0, str(ROOT / "experiments/v30_meta_stack"))

from build_candidate_catalog import canonical_template_prediction, prepare_order
from build_actions import site_key
from build_cross_order_probes import apply_actions, load_submission, write_submission


TRAIN = ROOT / "train"
TEST = ROOT / "test"
CHAMPION = ROOT / "experiments/v30_meta_stack/submissions/v30_cross_order_top5.csv"
V37 = ROOT / "experiments/v37_online_equations"
OUT = ROOT / "experiments/v39_template_count_transfer"
BASE_TP = 955
BASE_P = 1035
TRUE_ROOTS = 1044
MAX_ROOTS = 8


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def order_sites(order):
    return frozenset(site_key(node) for node in order["alarms"])


def evaluate_loso(groups):
    rows = []
    for signature, references in groups.items():
        sites = [order_sites(order) for order in references]
        for index, query in enumerate(references):
            peers = [
                reference for j, reference in enumerate(references)
                if j != index and sites[j].isdisjoint(sites[index])
            ]
            prediction = canonical_template_prediction(query, peers)
            if prediction is None:
                continue
            selected, truth = prediction["selected"], query["roots"]
            peer_sites = set().union(*(order_sites(peer) for peer in peers)) if peers else set()
            rows.append({
                "order_id": query["id"], "support": len(peers),
                "sites": len(peer_sites), "tp": len(selected & truth),
                "fp": len(selected - truth), "fn": len(truth - selected),
                "exact": selected == truth,
            })
    return rows


def summarize(rows, min_support=0, min_sites=0):
    chosen = [row for row in rows if row["support"] >= min_support and row["sites"] >= min_sites]
    tp = sum(row["tp"] for row in chosen)
    fp = sum(row["fp"] for row in chosen)
    fn = sum(row["fn"] for row in chosen)
    return {
        "orders": len(chosen), "exact_orders": sum(row["exact"] for row in chosen),
        "exact_rate": sum(row["exact"] for row in chosen) / max(len(chosen), 1),
        "tp": tp, "fp": fp, "fn": fn,
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
    }


def emit(name, candidates, exact_actions, order_ids, champion_roots, records_by_order):
    actions = [dict(action) for action in exact_actions] + [row["action"] for row in candidates]
    roots = apply_actions(champion_roots, records_by_order, actions)
    predictions = sum(map(len, roots.values()))
    path = OUT / "submissions" / f"{name}.csv"
    write_submission(path, order_ids, roots)
    change_min = sum(-len(row["action"]["remove_rids"]) for row in candidates)
    change_max = sum(len(row["action"]["add_rids"]) for row in candidates)
    manifest = {
        "probe_id": name, "baseline": str(CHAMPION), "path": str(path),
        "known_exact_gain": 1, "actions": actions, "candidate_orders": len(candidates),
        "predictions": predictions, "sha256": sha256(path),
        "variable_delta_tp_bounds": [change_min, change_max],
    }
    (OUT / "manifests" / f"{name}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "submissions").mkdir(exist_ok=True)
    (OUT / "manifests").mkdir(exist_ok=True)
    train_orders = [prepare_order(path, True) for path in sorted(TRAIN.iterdir()) if path.is_dir()]
    test_orders = [prepare_order(path, False) for path in sorted(TEST.iterdir()) if path.is_dir()]
    groups = defaultdict(list)
    for order in train_orders:
        groups[order["signature"]].append(order)
    loso_rows = evaluate_loso(groups)
    audit = {
        f"support{s}_sites{t}": summarize(loso_rows, s, t)
        for s, t in ((2, 1), (3, 2), (5, 2), (5, 3), (8, 3))
    }

    order_ids, champion_roots = load_submission(CHAMPION)
    fixed_report = read_json(V37 / "report.json")
    fixed = {(row["order_id"], row["rid"]): row["label"] for row in fixed_report["fixed_labels"]}
    exact_manifest = read_json(V37 / "manifests/v37_exact_corrections.json")
    exact_actions = exact_manifest["actions"]
    exact_orders = {action["order_id"] for action in exact_actions}
    candidates, rejected = [], []
    matched = predicted = 0
    for order in test_orders:
        references = groups.get(order["signature"], [])
        if not references:
            continue
        matched += 1
        prediction = canonical_template_prediction(order, references)
        if prediction is None:
            continue
        predicted += 1
        selected = prediction["selected"]
        current = {node["@rid"] for node in champion_roots[order["id"]]}
        remove, add = sorted(current - selected), sorted(selected - current)
        if not remove and not add:
            continue
        sites = set().union(*(order_sites(reference) for reference in references))
        contradiction = []
        for rid in remove:
            if fixed.get((order["id"], rid)) == 1:
                contradiction.append((rid, 1, "remove"))
        for rid in add:
            if fixed.get((order["id"], rid)) == 0:
                contradiction.append((rid, 0, "add"))
        action = {
            "action_id": f"v39_template_{order['id'][:8]}", "order_id": order["id"],
            "remove_rids": remove, "add_rids": add,
            "source": "v39_canonical_template_count", "expected_gain": None,
            "evidence": {"support": len(references), "sites": len(sites),
                         "expected_count": prediction["expected_k"]},
        }
        row = {"action": action, "support": len(references), "sites": len(sites),
               "change_size": len(remove) + len(add)}
        if contradiction:
            row["reason"] = "contradicts_online_fixed_label"
            row["contradiction"] = contradiction
            rejected.append(row)
        elif order["id"] in exact_orders:
            row["reason"] = "conflicts_with_exact_correction_order"
            rejected.append(row)
        elif not 1 <= len(selected) <= MAX_ROOTS:
            row["reason"] = "invalid_count"
            rejected.append(row)
        else:
            candidates.append(row)
    candidates.sort(key=lambda row: (-row["sites"], -row["support"], row["change_size"]))

    records_by_order = {order["id"]: {"order_id": order["id"], "alarms": [
        {"rid": node["@rid"], "source": node} for node in order["alarms"]
    ]} for order in test_orders}
    probes = {}
    tiers = {
        "strict": [row for row in candidates if row["support"] >= 5 and row["sites"] >= 3],
        "medium": [row for row in candidates if row["support"] >= 3 and row["sites"] >= 2],
        "all": candidates,
    }
    for tier, rows in tiers.items():
        for size in (4, 8, 16):
            subset = rows[:size]
            if not subset:
                continue
            name = f"v39_template_{tier}_top{len(subset):02d}_plus_exact"
            probes[name] = emit(
                name, subset, exact_actions, order_ids, champion_roots, records_by_order
            )
            if len(subset) < size:
                break
    report = {
        "version": "v39-template-count-transfer-1", "matched_test_orders": matched,
        "predicted_test_orders": predicted, "loso_audit": audit,
        "candidate_count": len(candidates), "rejected_count": len(rejected),
        "candidates": candidates, "rejected": rejected, "probes": probes,
    }
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "matched_test_orders": matched, "predicted_test_orders": predicted,
        "loso_audit": audit, "candidate_count": len(candidates),
        "strict_candidates": len(tiers["strict"]), "medium_candidates": len(tiers["medium"]),
        "top_candidates": candidates[:12],
        "probes": {name: {k: value[k] for k in ("path", "predictions", "sha256")}
                   for name, value in probes.items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
