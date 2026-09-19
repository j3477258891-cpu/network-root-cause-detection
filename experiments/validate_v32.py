"""Validate V32 probe artifacts and leaderboard TP reconstruction."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

from v32_delete_probes import (
    BASELINE_TP,
    CHAMPION,
    OUT,
    RECORDS,
    TRUE_ROOTS,
    historical_exclusions,
    load_submission,
    sha256,
)


EXPECTED = {
    "v32_delete_batch05": 5,
    "v32_delete_batch10": 10,
    "v32_delete_batch20": 20,
    "v32_delete_batch40": 40,
}


def recover_tp(score: float, predictions: int) -> int:
    return round(score * (TRUE_ROOTS + predictions) / 2)


def validate_csv(path: Path, valid_rids: dict[str, set[str]]):
    order_ids, roots = load_submission(path)
    if len(order_ids) != 546 or len(set(order_ids)) != 546:
        raise ValueError(f"bad order rows: {path}")
    total = 0
    for oid in order_ids:
        nodes = roots[oid]
        rids = [node["@rid"] for node in nodes]
        if not 1 <= len(rids) <= 8:
            raise ValueError(f"invalid root count: {oid}")
        if len(rids) != len(set(rids)):
            raise ValueError(f"duplicate root: {oid}")
        if not set(rids) <= valid_rids[oid]:
            raise ValueError(f"unknown RID: {oid}")
        total += len(rids)
    return order_ids, roots, total


def main():
    with gzip.open(RECORDS, "rt", encoding="utf-8") as handle:
        records = json.load(handle)["test"]
    valid_rids = {o["order_id"]: {a["rid"] for a in o["alarms"]} for o in records}
    base_ids, base_roots = load_submission(CHAMPION)
    touched, protected_in, protected_out = historical_exclusions()
    used_orders = set()
    result = {}

    for name, count in EXPECTED.items():
        manifest_path = OUT / "manifests" / f"{name}.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        path = Path(manifest["path"])
        order_ids, roots, predictions = validate_csv(path, valid_rids)
        if order_ids != base_ids:
            raise ValueError(f"order sequence changed: {name}")
        if predictions != 1035 - count or predictions != manifest["predictions"]:
            raise ValueError(f"prediction count mismatch: {name}")
        if sha256(path) != manifest["sha256"]:
            raise ValueError(f"hash mismatch: {name}")
        actions = manifest["actions"]
        if len(actions) != count:
            raise ValueError(f"action count mismatch: {name}")
        for action in actions:
            oid = action["order_id"]
            if oid in used_orders:
                raise ValueError(f"cross-batch order overlap: {oid}")
            used_orders.add(oid)
            if oid in touched:
                raise ValueError(f"historical order reused: {oid}")
            removed = set(action["remove_rids"])
            added = set(action["add_rids"])
            if len(removed) != 1 or added:
                raise ValueError(f"not an atomic deletion: {action['action_id']}")
            if any((oid, rid) in protected_in | protected_out for rid in removed):
                raise ValueError(f"protected node changed: {action['action_id']}")
            base = {n["@rid"] for n in base_roots[oid]}
            actual = {n["@rid"] for n in roots[oid]}
            if base - actual != removed or actual - base != added:
                raise ValueError(f"manifest diff mismatch: {action['action_id']}")

        changed_orders = {
            oid for oid in base_ids
            if {n["@rid"] for n in base_roots[oid]}
            != {n["@rid"] for n in roots[oid]}
        }
        if changed_orders != {a["order_id"] for a in actions}:
            raise ValueError(f"unexpected CSV changes: {name}")
        result[name] = {
            "actions": count,
            "predictions": predictions,
            "sha256": manifest["sha256"],
        }

    historical = [
        (0.906324, 1059, 953),
        (0.905373, 1059, 952),
        (0.918711, 1035, 955),
    ]
    for score, predictions, expected_tp in historical:
        if recover_tp(score, predictions) != expected_tp:
            raise ValueError((score, predictions, recover_tp(score, predictions)))

    print(json.dumps({
        "status": "ok",
        "baseline_tp": BASELINE_TP,
        "historical_tp_reconstruction": historical,
        "probes": result,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
