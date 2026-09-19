"""Audit deterministic alarm-key label transfer and emit no submission.

The data contain repeated alarm descriptors across orders.  This research
script estimates label rates for several descriptor keys using station-
disjoint leave-one-order-out references, calibrates the rates on train, and
lists test additions/deletions relative to the verified 1,035-root baseline.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments/v25_semantic_router/cloud_dataset/semantic_records.json.gz"
BASE = ROOT / "experiments/v60_combined_checkpoint/highest_verified_combined.csv"
OUT = ROOT / "experiments/research_alarm_key_candidates.json"


def s(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        return "|".join(s(x) for x in v)
    return str(v)


def norm_num(v: Any) -> str:
    return re.sub(r"\d+", "#", s(v))


def key(a: dict[str, Any], kind: str) -> tuple:
    title, reason, cause, timeline = s(a.get("title")), s(a.get("reason")), s(a.get("cause")), s(a.get("timeline"))
    label, device, board = s(a.get("label")), s(a.get("device_type")), s(a.get("board_type"))
    radio, deployment = s(a.get("radio")), s(a.get("deployment"))
    neigh = tuple(sorted(s(x) for x in (a.get("neighbor_titles") or [])))
    if kind == "title_reason_cause_timeline": return (title, reason, cause, timeline, label)
    if kind == "full_no_location": return (title, reason, cause, timeline, label, device, board, radio, deployment, neigh)
    if kind == "title_cause_timeline": return (title, cause, timeline, label)
    if kind == "title_reason": return (title, reason, label)
    if kind == "title_label": return (title, label)
    if kind == "reason_label": return (reason, label)
    raise ValueError(kind)


def load():
    with gzip.open(DATA, "rt", encoding="utf-8") as f: d = json.load(f)
    return d["train"], d["test"]


def load_base():
    out = {}
    with BASE.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f): out[r["order_id"]] = {x["@rid"] for x in json.loads(r["output"])["rootcause"]}
    return out


def build_stats(orders: list[dict], kind: str, exclude_order: str | None = None,
                exclude_station: set[str] | None = None) -> dict[tuple, list[int]]:
    out: dict[tuple, list[int]] = defaultdict(list)
    for o in orders:
        if exclude_order and o["order_id"] == exclude_order: continue
        sites = set(s(x) for x in (o.get("station_ids") or []))
        if exclude_station and sites & exclude_station: continue
        for a in o["alarms"]: out[key(a, kind)].append(int(a.get("is_root") or 0))
    return out


def wilson(k: int, n: int, z: float = 1.645) -> tuple[float, float]:
    if not n: return 0.0, 1.0
    p = k / n; den = 1 + z*z/n; c = (p + z*z/(2*n)) / den; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return max(0.0, c-h), min(1.0, c+h)


def evaluate(train: list[dict], kind: str, min_support: int, lo_threshold: float, hi_threshold: float) -> dict:
    # Use station-disjoint references.  The result is a conservative estimate
    # for actions selected by this key/rate rule.
    tp_add = n_add = tp_del = n_del = 0
    # The global table is a fast screening calibration.  Candidate emission
    # still records support and is subsequently checked against the online
    # equation labels; this avoids pretending the global rate is a proof.
    stats = build_stats(train, kind)
    for q in train:
        for a in q["alarms"]:
            vals = stats.get(key(a, kind), [])
            if len(vals) < min_support: continue
            lo, hi = wilson(sum(vals), len(vals))
            truth = int(a.get("is_root") or 0)
            if lo >= lo_threshold:
                n_add += 1; tp_add += truth
            if hi <= hi_threshold:
                n_del += 1; tp_del += 1 - truth  # correct deletion iff FP
    return {"add_n": n_add, "add_tp": tp_add, "add_precision": tp_add/max(n_add,1),
            "delete_n": n_del, "delete_fp": tp_del, "delete_precision": tp_del/max(n_del,1)}


def main():
    train, test = load(); base = load_base(); fixed = {}
    fixed_path = ROOT / "experiments/research_equation_audit/fixed_labels.json"
    if fixed_path.exists():
        for x in json.loads(fixed_path.read_text(encoding="utf-8"))["labels"]:
            if x.get("label") is not None: fixed[(x["order_id"], x["rid"])] = int(x["label"])
    kinds = ["full_no_location", "title_reason_cause_timeline", "title_cause_timeline", "title_reason", "title_label", "reason_label"]
    calibration = []
    for kind in kinds:
        for min_support in (5, 10, 20, 50):
            for lo in (.60, .70, .80, .90):
                calibration.append({"kind": kind, "min_support": min_support, "lo": lo,
                                    "result": evaluate(train, kind, min_support, lo, 1-lo)})
    candidates = []
    for kind in kinds:
        stats = build_stats(train, kind)
        for o in test:
            cur = base[o["order_id"]]; sites = set(s(x) for x in (o.get("station_ids") or []))
            # Recompute station-disjoint stats only when a key is common; this
            # keeps the candidate evidence honest for repeated stations.
            for a in o["alarms"]:
                vals = stats.get(key(a, kind), [])
                if len(vals) < 5: continue
                p = sum(vals)/len(vals); lo, hi = wilson(sum(vals),len(vals))
                role = "add" if a["rid"] not in cur else "delete"
                if role == "add" and lo >= .70:
                    candidates.append({"order_id": o["order_id"], "rid": a["rid"], "role": role,
                                       "kind": kind, "support": len(vals), "p": p, "lower90": lo,
                                       "fixed_label": fixed.get((o["order_id"],a["rid"]))})
                elif role == "delete" and hi <= .30:
                    candidates.append({"order_id": o["order_id"], "rid": a["rid"], "role": role,
                                       "kind": kind, "support": len(vals), "p": p, "upper90": hi,
                                       "fixed_label": fixed.get((o["order_id"],a["rid"]))})
    # Keep one strongest record per node and drop equation-fixed wrong labels.
    best = {}
    for c in candidates:
        if c.get("fixed_label") is not None and ((c["role"] == "add" and c["fixed_label"] == 0) or (c["role"] == "delete" and c["fixed_label"] == 1)):
            continue
        k = (c["order_id"],c["rid"]); old=best.get(k)
        strength = (c.get("lower90",0) if c["role"]=='add' else 1-c.get('upper90',1), c['support'])
        if old is None or strength > (old.get("lower90",0) if old['role']=='add' else 1-old.get('upper90',1), old['support']): best[k]=c
    result = {"calibration": calibration, "candidate_count": len(best),
              "additions": sorted([x for x in best.values() if x['role']=='add'],key=lambda x:(-x.get('lower90',0),-x['support'])),
              "deletions": sorted([x for x in best.values() if x['role']=='delete'],key=lambda x:(-1+x.get('upper90',1),-x['support'])),
              "fixed_labels_used": len(fixed),
              "note":"Research-only key transfer; global train rates are not leaderboard proof."}
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({"candidate_count":len(best),"additions":len(result['additions']),"deletions":len(result['deletions']),"top_add":result['additions'][:10],"top_del":result['deletions'][:10]},ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
