"""Research-only fuzzy template transfer audit.

This script never writes a submission file.  It uses the compact semantic
records (which contain train ``is_root`` labels) to test whether a relaxed
same-order template alignment can produce *new* baseline actions.  Every
rule is evaluated leave-one-order-out before test candidates are listed.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments/v25_semantic_router/cloud_dataset/semantic_records.json.gz"
BASE = ROOT / "experiments/v60_combined_checkpoint/highest_verified_combined.csv"
OUT = ROOT / "experiments/research_near_template.json"


def norm(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "|".join(norm(x) for x in value)
    return str(value)


def shape(value: Any) -> str:
    # Semantic records already use <NUM>, but also normalize raw values when
    # present in older exports.
    return re.sub(r"\d+", "#", norm(value))


def alarm_key(a: dict[str, Any], tier: str) -> tuple:
    # Keep the key deliberately interpretable.  Location is omitted in the
    # fuzzy tier; this is the only relaxation relative to V39.
    title, label = norm(a.get("title")), norm(a.get("label"))
    reason, cause = norm(a.get("reason")), norm(a.get("cause"))
    device, board = norm(a.get("device_type")), norm(a.get("board_type"))
    radio, deploy = norm(a.get("radio")), norm(a.get("deployment"))
    timeline = norm(a.get("timeline"))
    neigh = tuple(sorted(norm(x) for x in (a.get("neighbor_titles") or [])))
    if tier == "strict":
        return (title, label, shape(a.get("location")), reason, cause, device,
                board, radio, deploy, timeline, neigh)
    if tier == "medium":
        return (title, label, reason, cause, device, board, radio, deploy, neigh)
    if tier == "loose":
        return (title, label, cause, device, radio, deploy, neigh)
    raise ValueError(tier)


def order_sig(o: dict[str, Any], level: str) -> tuple:
    counts = Counter(norm(a.get("title")) for a in o["alarms"])
    targets = tuple(sorted(norm(a.get("title")) for a in o["alarms"]
                          if norm(a.get("label")) == "TargetAlarm"))
    if level == "exact":
        return (tuple(sorted(counts.items())), targets, len(o["alarms"]))
    if level == "title_count":
        return (tuple(sorted(counts.items())), len(o["alarms"]))
    if level == "target_count":
        return (targets, len(o["alarms"]))
    raise ValueError(level)


def load() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    with gzip.open(DATA, "rt", encoding="utf-8") as f:
        d = json.load(f)
    return d["train"], d["test"]


def load_base() -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    with BASE.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            result[row["order_id"]] = {x["@rid"] for x in json.loads(row["output"])["rootcause"]}
    return result


def build_ref_index(train: list[dict[str, Any]]) -> dict[str, dict[tuple, list[dict[str, Any]]]]:
    return {level: _index_level(train, level) for level in ("exact", "title_count", "target_count")}


def _index_level(train: list[dict[str, Any]], level: str) -> dict[tuple, list[dict[str, Any]]]:
    result: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for o in train:
        result[order_sig(o, level)].append(o)
    return result


def refs_for(query: dict[str, Any], ref_index: dict[str, dict[tuple, list[dict[str, Any]]]], level: str,
             min_refs: int, max_refs: int = 50) -> list[dict[str, Any]]:
    sig = order_sig(query, level)
    refs = [o for o in ref_index[level].get(sig, []) if o["order_id"] != query["order_id"]]
    # Deterministic station disjointness is important: a station can repeat
    # the same alarm pattern and otherwise creates an optimistic leak.
    qsites = set(norm(x) for x in (query.get("station_ids") or []))
    disjoint = [o for o in refs if qsites.isdisjoint(set(norm(x) for x in (o.get("station_ids") or [])))]
    if len(disjoint) >= min_refs:
        refs = disjoint
    return refs[:max_refs]


def predict(query: dict[str, Any], refs: list[dict[str, Any]], tier: str,
            rate_threshold: float, min_support: int, count_tolerance: int = 0) -> dict[str, Any] | None:
    if len(refs) < min_support:
        return None
    votes: dict[tuple, list[int]] = defaultdict(list)
    for ref in refs:
        for a in ref["alarms"]:
            votes[alarm_key(a, tier)].append(int(a.get("is_root") or 0))
    selected: set[str] = set()
    confidence: dict[str, float] = {}
    support: dict[str, int] = {}
    for a in query["alarms"]:
        vals = votes.get(alarm_key(a, tier), [])
        if len(vals) < min_support:
            continue
        p = sum(vals) / len(vals)
        if p >= rate_threshold:
            selected.add(a["rid"])
            confidence[a["rid"]] = p
            support[a["rid"]] = len(vals)
    # A template transfer is only useful when it predicts a stable count.
    ks = [sum(int(a.get("is_root") or 0) for a in ref["alarms"]) for ref in refs]
    mode_k, mode_n = Counter(ks).most_common(1)[0]
    if abs(len(selected) - mode_k) > count_tolerance:
        return None
    return {"selected": selected, "mode_k": mode_k, "mode_support": mode_n,
            "refs": len(refs), "confidence": confidence, "support": support}


def eval_rule(train: list[dict[str, Any]], ref_index: dict[str, dict[tuple, list[dict[str, Any]]]], level: str, tier: str, threshold: float,
              min_refs: int, count_tolerance: int) -> dict[str, Any]:
    tp = fp = fn = exact = eligible = 0
    deltas: list[int] = []
    for q in train:
        refs = refs_for(q, ref_index, level, min_refs)
        pred = predict(q, refs, tier, threshold, min_refs, count_tolerance)
        if pred is None:
            continue
        truth = {a["rid"] for a in q["alarms"] if int(a.get("is_root") or 0)}
        s = pred["selected"]
        eligible += 1; tp += len(s & truth); fp += len(s - truth); fn += len(truth - s)
        exact += int(s == truth); deltas.append(len(s & truth) - len(s - truth))
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return {"eligible": eligible, "exact": exact, "exact_rate": exact / max(eligible, 1),
            "tp": tp, "fp": fp, "fn": fn, "f1": f1,
            "delta_mean": sum(deltas) / max(len(deltas), 1),
            "delta_positive": sum(x > 0 for x in deltas),
            "delta_nonnegative": sum(x >= 0 for x in deltas)}


def test_candidates(train: list[dict[str, Any]], ref_index: dict[str, dict[tuple, list[dict[str, Any]]]], test: list[dict[str, Any]], base: dict[str, set[str]],
                    level: str, tier: str, threshold: float, min_refs: int, count_tolerance: int) -> list[dict[str, Any]]:
    out = []
    for q in test:
        refs = refs_for(q, ref_index, level, min_refs)
        pred = predict(q, refs, tier, threshold, min_refs, count_tolerance)
        if pred is None:
            continue
        cur = base.get(q["order_id"], set())
        add = sorted(pred["selected"] - cur); rem = sorted(cur - pred["selected"])
        if not add and not rem:
            continue
        # Only equal-count one-for-one changes are emitted as candidate pairs.
        if len(add) != len(rem):
            continue
        for rr, aa in zip(rem, add):
            out.append({"order_id": q["order_id"], "remove_rid": rr, "add_rid": aa,
                        "refs": pred["refs"], "mode_k": pred["mode_k"],
                        "remove_confidence": 1.0 - pred["confidence"].get(rr, 0.0),
                        "add_confidence": pred["confidence"].get(aa, 0.0),
                        "remove_support": pred["support"].get(rr, 0),
                        "add_support": pred["support"].get(aa, 0),
                        "level": level, "tier": tier})
    return out


def main() -> None:
    train, test = load(); base = load_base(); ref_index = build_ref_index(train)
    rules = []
    for level in ("exact", "title_count", "target_count"):
        for tier in ("strict", "medium", "loose"):
            for threshold in (0.8, 0.9, 1.0):
                for min_refs in (2, 3, 5):
                    result = eval_rule(train, ref_index, level, tier, threshold, min_refs, 0)
                    rules.append({"level": level, "tier": tier, "threshold": threshold,
                                  "min_refs": min_refs, "result": result})
    # Rank rules by a conservative lower bound on per-order positive delta.
    def rank(x: dict[str, Any]) -> tuple:
        r = x["result"]
        # Wilson-like crude lower bound; require at least 10 eligible orders.
        n = r["eligible"]; pos = r["delta_nonnegative"]
        lb = (pos / n - 1.96 * math.sqrt(max((pos / n) * (1 - pos / n) / max(n, 1), 0))) if n else -1
        return (lb, r["exact_rate"], r["f1"], n)
    ranked = sorted(rules, key=rank, reverse=True)
    candidates = []
    for rule in ranked[:30]:
        candidates.extend(test_candidates(train, ref_index, test, base, rule["level"], rule["tier"],
                                           rule["threshold"], rule["min_refs"], 0))
    # De-duplicate candidate pairs, retaining the strongest rule evidence.
    by: dict[tuple[str, str, str], dict[str, Any]] = {}
    for c in candidates:
        key = (c["order_id"], c["remove_rid"], c["add_rid"])
        old = by.get(key)
        if old is None or (c["add_confidence"] + c["remove_confidence"], c["refs"]) > (old["add_confidence"] + old["remove_confidence"], old["refs"]):
            by[key] = c
    output = {"rules": ranked, "top_rules": ranked[:30], "candidate_count": len(by),
              "candidates": sorted(by.values(), key=lambda x: (-x["add_confidence"] - x["remove_confidence"], -x["refs"], x["order_id"]))}
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"train_orders": len(train), "test_orders": len(test),
                      "candidate_count": len(by), "top_rules": ranked[:10]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
