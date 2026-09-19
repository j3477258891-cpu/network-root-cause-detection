"""Resolve ambiguous exact-template matches with transferred root counts.

V39 proved that canonical root patterns are nearly deterministic across sites,
but rejected a test order whenever multiple alarms shared the same canonical
key.  Here the template supplies the exact root count per canonical key and a
cross-fitted V30 score ranks only the otherwise indistinguishable nodes.
"""

from __future__ import annotations

import csv
import gzip
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(r"D:\zgyidong")
EXP = ROOT / "experiments"
for value in (EXP, EXP / "v27_closed_loop", EXP / "v30_meta_stack"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import numpy as np

from build_candidate_catalog import canonical_key, prepare_order
from build_actions import site_key
from v30_meta_stack import exact_count_mask


TRAIN = ROOT / "train"
TEST = ROOT / "test"
RECORDS = EXP / "v25_semantic_router/cloud_dataset/semantic_records.json.gz"
V30 = EXP / "v30_meta_stack"
BASE = EXP / "v60_combined_checkpoint/highest_verified_combined.csv"
OUT = EXP / "v79_ambiguous_template_alignment"
SEEDS = (20260801, 20260817, 20260831)


def order_sites(order: dict) -> frozenset:
    return frozenset(site_key(node) for node in order["alarms"])


def stable_pattern(references: list[dict], min_support: int = 2):
    if len(references) < min_support:
        return None
    patterns = []
    for reference in references:
        counts = Counter(
            canonical_key(node) for node in reference["alarms"]
            if node["@rid"] in reference["roots"]
        )
        patterns.append(tuple(sorted(counts.items())))
    pattern, support = Counter(patterns).most_common(1)[0]
    if support / len(patterns) < 0.90:
        return None
    for seed in SEEDS:
        rng = random.Random(seed)
        sample = [patterns[rng.randrange(len(patterns))] for _ in patterns]
        if Counter(sample).most_common(1)[0][0] != pattern:
            return None
    return Counter(dict(pattern)), support


def predict(query: dict, references: list[dict], scores: dict[tuple[str, str], float]):
    result = stable_pattern(references)
    if result is None:
        return None
    pattern, support = result
    by_key = defaultdict(list)
    for node in query["alarms"]:
        by_key[canonical_key(node)].append(node["@rid"])
    selected = set()
    ambiguous_keys = 0
    for key, count in pattern.items():
        matches = by_key.get(key, [])
        if len(matches) < count:
            return None
        if len(matches) > count:
            ambiguous_keys += 1
        ranked = sorted(
            matches,
            key=lambda rid: (-scores.get((query["id"], rid), -1e30), rid),
        )
        selected.update(ranked[:count])
    return {"selected": selected, "support": support,
            "ambiguous_keys": ambiguous_keys, "expected_k": len(selected)}


def score_lookup(split: str, orders: list[dict]) -> dict[tuple[str, str], float]:
    path = V30 / f"v30_consensus_{'oof' if split == 'train' else 'test'}.npy"
    values = np.load(path)
    with gzip.open(RECORDS, "rt", encoding="utf-8") as handle:
        records = json.load(handle)[split]
    flattened = [(order["order_id"], alarm["rid"])
                 for order in records for alarm in order["alarms"]]
    if len(flattened) != len(values):
        raise RuntimeError((split, len(flattened), len(values)))
    return {key: float(value) for key, value in zip(flattened, values)}


def load_base() -> dict[str, set[str]]:
    result = {}
    with BASE.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            result[row["order_id"]] = {
                node["@rid"] for node in json.loads(row["output"])["rootcause"]
            }
    return result


def metrics(rows: list[dict], field: str) -> dict:
    tp = fp = fn = exact = 0
    for row in rows:
        chosen, truth = row[field], row["truth"]
        tp += len(chosen & truth)
        fp += len(chosen - truth)
        fn += len(truth - chosen)
        exact += chosen == truth
    return {
        "orders": len(rows), "tp": tp, "fp": fp, "fn": fn,
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "exact_orders": exact, "exact_rate": exact / max(len(rows), 1),
    }


def main() -> None:
    train = [prepare_order(path, True) for path in sorted(TRAIN.iterdir()) if path.is_dir()]
    test = [prepare_order(path, False) for path in sorted(TEST.iterdir()) if path.is_dir()]
    groups = defaultdict(list)
    for order in train:
        groups[order["signature"]].append(order)
    train_scores = score_lookup("train", train)
    test_scores = score_lookup("test", test)

    # Cross-site leave-one-order-out audit.  The baseline uses the same V30 OOF
    # score but keeps the truth count per order, isolating alignment quality.
    audit_rows = []
    for query in train:
        peers = [reference for reference in groups[query["signature"]]
                 if reference["id"] != query["id"]
                 and order_sites(reference).isdisjoint(order_sites(query))]
        prediction = predict(query, peers, train_scores)
        if prediction is None:
            continue
        k = len(query["roots"])
        ranked_all = sorted(
            [node["@rid"] for node in query["alarms"]],
            key=lambda rid: (-train_scores[(query["id"], rid)], rid),
        )
        audit_rows.append({
            "order_id": query["id"], "truth": query["roots"],
            "template": prediction["selected"],
            "v30_oracle_count": set(ranked_all[:k]),
            "ambiguous_keys": prediction["ambiguous_keys"],
            "support": prediction["support"],
        })

    base = load_base()
    test_rows = []
    for query in test:
        peers = groups.get(query["signature"], [])
        prediction = predict(query, peers, test_scores)
        if prediction is None:
            continue
        current = base[query["id"]]
        test_rows.append({
            "order_id": query["id"], "current": sorted(current),
            "selected": sorted(prediction["selected"]),
            "remove": sorted(current - prediction["selected"]),
            "add": sorted(prediction["selected"] - current),
            "support": prediction["support"],
            "ambiguous_keys": prediction["ambiguous_keys"],
            "expected_k": prediction["expected_k"],
        })

    changed = [row for row in test_rows if row["remove"] or row["add"]]
    ambiguous_audit = [row for row in audit_rows if row["ambiguous_keys"]]
    report = {
        "version": "v79-ambiguous-template-alignment-1",
        "train_orders": len(train), "test_orders": len(test),
        "audit": {
            "all_template_orders": {
                "template": metrics(audit_rows, "template"),
                "v30_oracle_count": metrics(audit_rows, "v30_oracle_count"),
            },
            "ambiguous_template_orders": {
                "template": metrics(ambiguous_audit, "template"),
                "v30_oracle_count": metrics(ambiguous_audit, "v30_oracle_count"),
            },
        },
        "test_predicted_orders": len(test_rows),
        "test_ambiguous_orders": sum(row["ambiguous_keys"] > 0 for row in test_rows),
        "test_changed_orders": len(changed),
        "test_total_removes": sum(len(row["remove"]) for row in changed),
        "test_total_adds": sum(len(row["add"]) for row in changed),
        "changed": changed,
        "warning": "OOF audit uses true per-order count for the V30 comparator; test labels remain unknown.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "changed"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
