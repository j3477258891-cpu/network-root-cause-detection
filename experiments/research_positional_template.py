"""Research-only positional template transfer audit.

This deliberately does not emit a submission.  It aligns alarms from test
orders with same-signature train orders using progressively relaxed semantic
keys, evaluates the rule leave-one-order-out on train, and reports possible
baseline swaps for test orders.
"""

from __future__ import annotations

import csv
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments/v25_semantic_router/cloud_dataset/semantic_records.json.gz"
BASE = ROOT / "experiments/v60_combined_checkpoint/highest_verified_combined.csv"
OUT = ROOT / "experiments/research_positional_template.json"


def norm(v):
    if v is None:
        return ""
    if isinstance(v, list):
        return "|".join(norm(x) for x in v)
    return re.sub(r"\d+", "#", str(v)).strip().lower()


def signature(order):
    # Ignore target labels in the query signature; they are not available at
    # test time and title/count is the stable part of the template.
    return tuple((norm(k), int(v)) for k, v in order["signature"][0]), len(order["alarms"])


def alarm_key(a, level):
    title = norm(a.get("title")); label = norm(a.get("label"))
    cause = norm(a.get("cause")); device = norm(a.get("device_type"))
    board = norm(a.get("board_type")); radio = norm(a.get("radio"))
    deploy = norm(a.get("deployment")); reason = norm(a.get("reason"))
    neigh = tuple(sorted(norm(x) for x in (a.get("neighbor_titles") or [])))
    if level == "full":
        return (title, label, reason, cause, device, board, radio, deploy, neigh)
    if level == "device":
        return (title, label, cause, device, board, radio, deploy)
    if level == "title_cause":
        return (title, label, cause)
    if level == "title":
        return (title, label)
    raise ValueError(level)


def load_base():
    out = {}
    with BASE.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            out[row["order_id"]] = {
                x["@rid"] for x in json.loads(row["output"])["rootcause"]
            }
    return out


def refs_for(order, groups, exclude_id=None):
    refs = [x for x in groups.get(signature(order), []) if x["order_id"] != exclude_id]
    sites = set(norm(x) for x in (order.get("station_ids") or []))
    disjoint = [x for x in refs if sites.isdisjoint(set(norm(y) for y in (x.get("station_ids") or [])))]
    return disjoint if len(disjoint) >= 1 else refs


def predict(order, refs, level, min_votes=1):
    votes = defaultdict(list)
    for ref in refs:
        for a in ref["alarms"]:
            votes[alarm_key(a, level)].append(int(a.get("is_root") or 0))
    pred = set(); confidence = {}
    for a in order["alarms"]:
        vals = votes.get(alarm_key(a, level), [])
        if len(vals) < min_votes:
            continue
        p = sum(vals) / len(vals)
        confidence[a["rid"]] = (p, len(vals))
        if p >= 0.5:
            pred.add(a["rid"])
    return pred, confidence


def evaluate(train, groups, level, min_refs, min_votes):
    rows = []
    for q in train:
        refs = refs_for(q, groups, q["order_id"])
        if len(refs) < min_refs:
            continue
        pred, _ = predict(q, refs, level, min_votes)
        truth = {a["rid"] for a in q["alarms"] if int(a.get("is_root") or 0)}
        rows.append(len(pred & truth) - len(pred - truth))
    return {
        "n": len(rows),
        "positive": sum(x > 0 for x in rows),
        "nonnegative": sum(x >= 0 for x in rows),
        "mean_delta": sum(rows) / max(len(rows), 1),
        "distribution": dict(sorted(Counter(rows).items())),
    }


def test_candidates(test, groups, base, level, min_refs, min_votes):
    out = []
    for q in test:
        refs = refs_for(q, groups)
        if len(refs) < min_refs:
            continue
        pred, confidence = predict(q, refs, level, min_votes)
        cur = base[q["order_id"]]
        add = sorted(pred - cur); remove = sorted(cur - pred)
        # Equal-count actions are the only ones retained for a same-P probe.
        if not add or not remove or len(add) != len(remove):
            continue
        # Pair highest-confidence additions with lowest-confidence deletions.
        add.sort(key=lambda rid: confidence.get(rid, (0, 0))[0], reverse=True)
        remove.sort(key=lambda rid: confidence.get(rid, (0, 0))[0])
        for rr, aa in zip(remove, add):
            out.append({
                "order_id": q["order_id"], "remove_rid": rr, "add_rid": aa,
                "refs": len(refs), "level": level,
                "remove_p": confidence.get(rr, (0, 0))[0],
                "add_p": confidence.get(aa, (0, 0))[0],
                "margin": confidence.get(aa, (0, 0))[0] - confidence.get(rr, (0, 0))[0],
            })
    return out


def main():
    with gzip.open(DATA, "rt", encoding="utf-8") as f:
        data = json.load(f)
    train, test = data["train"], data["test"]
    groups = defaultdict(list)
    for order in train:
        groups[signature(order)].append(order)
    base = load_base()
    rules = []
    candidates = []
    for level in ("full", "device", "title_cause", "title"):
        for min_refs in (1, 2, 3, 5):
            for min_votes in (1, 2):
                ev = evaluate(train, groups, level, min_refs, min_votes)
                cs = test_candidates(test, groups, base, level, min_refs, min_votes)
                rules.append({"level": level, "min_refs": min_refs, "min_votes": min_votes,
                              "oof": ev, "test_candidates": len(cs)})
                candidates.extend(cs)
    unique = {}
    for c in candidates:
        key = (c["order_id"], c["remove_rid"], c["add_rid"])
        old = unique.get(key)
        if old is None or (c["margin"], c["refs"]) > (old["margin"], old["refs"]):
            unique[key] = c
    result = {"version": 1, "rules": rules,
              "candidate_count": len(unique),
              "candidates": sorted(unique.values(), key=lambda x: (-x["margin"], -x["refs"], x["order_id"])),
              "interpretation": "Research only; no submission emitted."}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"candidate_count": len(unique), "top_rules": sorted(rules, key=lambda x: (x["oof"]["mean_delta"], x["oof"]["nonnegative"], x["oof"]["n"]), reverse=True)[:10], "top_candidates": result["candidates"][:20]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
