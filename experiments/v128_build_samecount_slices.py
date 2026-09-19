"""Emit top-K slices of the V128 same-count rerank by score improvement."""
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
BASE = EXP / "v124_campaign/v124_single_pair_015_from_probe13.csv"
FULL = EXP / "v128_v30_rerank/v128_v30_samecount_rerank_from_pair015.csv"
RECORDS = EXP / "v25_semantic_router/cloud_dataset/semantic_records.json.gz"
SCORES = EXP / "v30_meta_stack/v30_consensus_test.npy"
OUT = EXP / "v128_v30_rerank"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path):
    out, order_ids = {}, []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            order_ids.append(r["order_id"])
            out[r["order_id"]] = r["output"]
    return order_ids, out


def ids(raw: str) -> set[str]:
    return {x["@rid"] for x in json.loads(raw)["rootcause"]}


def main() -> None:
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    full_path = Path(sys.argv[2]) if len(sys.argv) > 2 else FULL
    score_path = Path(sys.argv[3]) if len(sys.argv) > 3 else SCORES
    out_dir = Path(sys.argv[4]) if len(sys.argv) > 4 else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    order_ids, base = load(BASE)
    _, full = load(full_path)
    with gzip.open(RECORDS, "rt", encoding="utf-8") as f:
        records = json.load(f)["test"]
    with np.load(EXP / "v25_semantic_router/cloud_dataset/v25_semantic_router.npz") as z:
        ptr = z["test_alarm_ptr"]
    scores = np.load(score_path)
    alarm_scores = {}
    for i, rec in enumerate(records):
        for j, a in enumerate(rec["alarms"]):
            alarm_scores[(rec["order_id"], a["rid"])] = float(scores[int(ptr[i]) + j])
    proposals = []
    for oid in order_ids:
        old, new = ids(base[oid]), ids(full[oid])
        if old == new:
            continue
        old_score = sum(alarm_scores.get((oid, r), 0.0) for r in old)
        new_score = sum(alarm_scores.get((oid, r), 0.0) for r in new)
        proposals.append({"order_id": oid, "removed": sorted(old - new), "added": sorted(new - old),
                          "score_gain": new_score - old_score})
    proposals.sort(key=lambda x: x["score_gain"], reverse=True)
    chosen = proposals[:k]
    out = dict(base)
    for c in chosen:
        out[c["order_id"]] = full[c["order_id"]]
    total = sum(len(json.loads(raw)["rootcause"]) for raw in out.values())
    if total != 1035:
        raise RuntimeError(f"P changed to {total}")
    prefix = "v129_station" if "station_extra" in score_path.name else "v128_v30"
    stem = f"{prefix}_samecount_top{k}_from_pair015"
    path = out_dir / f"{stem}.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["order_id", "output"])
        w.writeheader()
        for oid in order_ids:
            w.writerow({"order_id": oid, "output": out[oid]})
    meta = {"version": "v128-samecount-rerank-slice", "base": str(BASE),
            "base_sha256": sha256(BASE), "full_rerank": str(full_path),
            "full_rerank_sha256": sha256(full_path), "source_scores": str(score_path),
            "base_public_f1": 0.921597, "base_tp": 958, "slice_size": k,
            "available_changed_orders": len(proposals), "selected": chosen,
            "predictions": total, "orders": len(order_ids), "sha256": sha256(path),
            "status": "exploratory_not_scored",
            "warning": "Score-gain ranking is model-derived, not online evidence."}
    (out_dir / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({x: meta[x] for x in ["version", "slice_size", "available_changed_orders", "predictions", "orders", "sha256"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
