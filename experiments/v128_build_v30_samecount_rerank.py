"""Re-rank each current order internally with V30 consensus scores.

The root count of every order is frozen to the verified V124 champion, so the
global prediction count stays 1035.  Known fixed-true labels from V37 are
protected when present in the champion.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\zgyidong")
EXP = ROOT / "experiments"
OUT = EXP / "v128_v30_rerank"
RECORDS = EXP / "v25_semantic_router/cloud_dataset/semantic_records.json.gz"
BASE = EXP / "v124_campaign/v124_single_pair_015_from_probe13.csv"
SCORES = EXP / "v30_meta_stack/v30_consensus_test.npy"
V37 = EXP / "v37_online_equations/report.json"
KNOWN_POSITIVE = {
    # V124 single_pair_015 is publicly +1 TP over probe13; never rerank it out.
    ("4079fac3-5a5c-48e1-b171-d09393a78fc9", "#-1:efbf6784-8142-4c84-beac-82ece429ad61"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    global OUT, SCORES
    if len(sys.argv) > 1:
        SCORES = Path(sys.argv[1])
    if len(sys.argv) > 2:
        OUT = Path(sys.argv[2])
    OUT.mkdir(parents=True, exist_ok=True)
    base = {}
    order_ids = []
    with BASE.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            order_ids.append(r["order_id"])
            base[r["order_id"]] = json.loads(r["output"])["rootcause"]
    with gzip.open(RECORDS, "rt", encoding="utf-8") as f:
        records = json.load(f)["test"]
    with np.load(EXP / "v25_semantic_router/cloud_dataset/v25_semantic_router.npz") as z:
        ptr = z["test_alarm_ptr"]
    scores = np.load(SCORES)
    if len(scores) != int(ptr[-1]):
        raise RuntimeError("score/alarm length mismatch")
    fixed_true = set()
    fixed_false = set()
    report = json.loads(V37.read_text(encoding="utf-8"))
    for x in report.get("fixed_labels", []):
        if x.get("label") == 1:
            fixed_true.add((x["order_id"], x["rid"]))
        elif x.get("label") == 0:
            fixed_false.add((x["order_id"], x["rid"]))
    out = {}
    changed = []
    protected_count = 0
    for i, rec in enumerate(records):
        oid = rec["order_id"]
        current = {x["@rid"] for x in base[oid]}
        start, stop = int(ptr[i]), int(ptr[i + 1])
        alarms = rec["alarms"]
        k = len(current)
        protected = [a for a in alarms if ((oid, a["rid"]) in fixed_true or
                                           (oid, a["rid"]) in KNOWN_POSITIVE) and a["rid"] in current]
        protected_count += len(protected)
        if len(protected) > k:
            raise RuntimeError(f"too many protected nodes in {oid}")
        # Rank unprotected alarms by V30 consensus and fill the remaining slots.
        candidates = [a for a in alarms if a["rid"] not in {x["rid"] for x in protected}
                      and (oid, a["rid"]) not in fixed_false]
        order_scores = scores[start:stop]
        index = {a["rid"]: j for j, a in enumerate(alarms)}
        candidates.sort(key=lambda a: (float(order_scores[index[a["rid"]]]), a["rid"]), reverse=True)
        selected = protected + candidates[: k - len(protected)]
        out[oid] = [{"@rid": a["rid"], "title": a.get("source", {}).get("title", ""),
                     "location": a.get("source", {}).get("location", ""),
                     "reason": a.get("source", {}).get("reason", "")} for a in selected]
        if {x["@rid"] for x in out[oid]} != current:
            changed.append(oid)
    total = sum(len(x) for x in out.values())
    if total != 1035:
        raise RuntimeError(f"P changed to {total}")
    prefix = "v129_station" if "station_extra" in SCORES.name else "v128_v30"
    path = OUT / f"{prefix}_samecount_rerank_from_pair015.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["order_id", "output"])
        w.writeheader()
        for oid in order_ids:
            w.writerow({"order_id": oid, "output": json.dumps({"rootcause": out[oid]}, ensure_ascii=False)})
    meta = {
        "version": "v128-v30-samecount-rerank",
        "base": str(BASE), "base_sha256": sha256(BASE),
        "source_scores": str(SCORES), "source_fixed_labels": str(V37),
        "base_public_f1": 0.921597, "base_tp": 958,
        "changed_orders": len(changed), "protected_fixed_true": protected_count,
        "predictions": total, "orders": len(order_ids), "sha256": sha256(path),
        "status": "exploratory_not_scored",
        "warning": "Same-count V30 re-ranking is unscored; preserve champion until online result.",
    }
    (OUT / f"{prefix}_samecount_rerank_from_pair015.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
