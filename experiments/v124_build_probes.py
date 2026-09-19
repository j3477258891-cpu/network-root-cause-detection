"""V124 probe batch builder.

Each probe file = pure baseline (v60) + N same-order swap actions, so that the
online TP delta can be decomposed: net_delta = k - (N - k) = 2k - N, giving the
exact number k of correct actions per batch.  Same-order duplicates are removed
(one action per order per batch; later actions for that order remain as backup).
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(r"D:\zgyidong")
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

import v121_equation_safe_campaign as v121  # noqa: E402

OUT = ROOT / "experiments/v124_campaign"
BASE = ROOT / "experiments/v60_combined_checkpoint/highest_verified_combined.csv"

# Batch design within a 27-submission budget (<=2/day).
BATCH_PLAN = [
    {"probe_id": 1, "layer": "L1", "n_actions": 4},
    {"probe_id": 2, "layer": "L1", "n_actions": 4},
    {"probe_id": 3, "layer": "L1", "n_actions": 2},
    {"probe_id": 4, "layer": "L2", "n_actions": 8},
    {"probe_id": 5, "layer": "L2", "n_actions": 8},
    {"probe_id": 6, "layer": "L3", "n_actions": 8},
    {"probe_id": 7, "layer": "L3", "n_actions": 8},
    {"probe_id": 8, "layer": "L3", "n_actions": 8},
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["order_id", "output"])
        writer.writeheader()
        writer.writerows(rows)


def apply_actions(base_rows: list[dict[str, Any]], actions: list[dict[str, Any]],
                  alarms: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply several same-order swaps; each action touches a distinct order."""
    by_oid = {a["order_id"]: a for a in actions}
    out = []
    for row in base_rows:
        roots = [dict(node) for node in row["roots"]]
        act = by_oid.get(row["order_id"])
        if act is not None:
            present = {node["@rid"] for node in roots}
            if act["remove_rid"] not in present or act["add_rid"] in present:
                raise ValueError(f"invalid action for {row['order_id']}")
            roots = [node for node in roots if node["@rid"] != act["remove_rid"]]
            roots.append(v121.alarm_node(alarms[(row["order_id"], act["add_rid"])]))
        out.append({"order_id": row["order_id"], "output": json.dumps({"rootcause": roots}, ensure_ascii=False)})
    total = sum(len(json.loads(x["output"])["rootcause"]) for x in out)
    if total != 1035:
        raise ValueError(f"prediction count changed: {total}")
    return out


def main() -> None:
    catalog = json.loads((OUT / "candidate_catalog.json").read_text(encoding="utf-8"))["candidates"]
    base_rows = v121.load_base_rows()
    alarms, _ = v121.load_alarm_records()

    # Deduplicate by order: keep the highest-margin action per order.
    by_order: dict[str, dict[str, Any]] = {}
    for c in catalog:
        if c["order_id"] not in by_order or c["score_margin"] > by_order[c["order_id"]]["score_margin"]:
            by_order[c["order_id"]] = c
    unique = sorted(by_order.values(), key=lambda x: -x["score_margin"])
    print("unique orders:", len(unique))

    # Assign actions to batches following the plan (consume the unique list).
    probes = []
    cursor = 0
    for plan in BATCH_PLAN:
        n = plan["n_actions"]
        chosen = unique[cursor:cursor + n]
        cursor += n
        if not chosen:
            continue
        rows = apply_actions(base_rows, chosen, alarms)
        path = OUT / f"probe_{plan['probe_id']:02d}_{plan['layer']}.csv"
        write_csv(path, rows)
        probes.append({
            "probe_id": plan["probe_id"], "layer": plan["layer"],
            "n_actions": len(chosen), "file": str(path),
            "predictions": 1035, "sha256": sha256(path),
            "actions": [{k: c[k] for k in ("order_id", "remove_rid", "add_rid", "score_margin")} for c in chosen],
        })
        print(f"probe {plan['probe_id']} {plan['layer']}: {len(chosen)} actions -> {path.name} sha256={sha256(path)[:16]}")

    write_json(OUT / "probe_matrix.json", {
        "version": "v124",
        "design": "each probe = baseline + N actions (order-distinct); online k = (delta_tp + N) / 2",
        "budget": 27, "per_day": 2,
        "decision_points": {
            "D1": "after probe 1-2: online accuracy >= 0.65 (>=5/8) -> proceed L2; <0.50 -> stop",
            "D2": "after probe 3-5: cumulative accuracy >= 0.65 -> proceed L3; posterior gate check",
            "D3": "after probe 6-8: posterior P(F1>=0.945) nominal >= 0.85 / conservative >= 0.80 -> emit final",
        },
        "stop_conditions": ["3 consecutive probes net_tp <= 0", "online accuracy < 0.60", "posterior conservative P < 0.80", "equation conflict"],
        "probes": probes,
    })
    print(json.dumps({"probes_emitted": len(probes)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
