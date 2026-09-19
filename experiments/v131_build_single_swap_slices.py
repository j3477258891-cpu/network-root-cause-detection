"""Emit station-rerank slices restricted to one-for-one swaps.

This is a deliberately separate route from V129's multi-node same-count
rerank: it keeps only orders where the reranker changes exactly one root.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(r"D:\zgyidong")
EXP = ROOT / "experiments"
BASE = EXP / "v124_campaign/v124_single_pair_015_from_probe13.csv"
FULL = EXP / "v129_station_rerank/v129_station_samecount_rerank_from_pair015.csv"
RECORDS = EXP / "v25_semantic_router/cloud_dataset/semantic_records.json.gz"
SCORES = EXP / "v30_meta_stack/station_extra_trees_test.npy"
OUT = EXP / "v131_single_swap"
DATA = EXP / "v25_semantic_router/cloud_dataset/v25_semantic_router.npz"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    out, order_ids = {}, []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            order_ids.append(row["order_id"])
            out[row["order_id"]] = row["output"]
    return order_ids, out


def ids(raw: str) -> set[str]:
    return {x["@rid"] for x in json.loads(raw)["rootcause"]}


def exact_count_mask(scores: np.ndarray, ptr: np.ndarray, target: int) -> np.ndarray:
    selected = np.zeros(len(scores), dtype=bool)
    optional = []
    for start, stop in zip(ptr[:-1], ptr[1:]):
        start, stop = int(start), int(stop)
        ranked = np.argsort(-scores[start:stop], kind="stable")[:8]
        selected[start + ranked[0]] = True
        optional.extend((start + ranked[1:]).tolist())
    optional = np.asarray(optional, dtype=np.int64)
    optional = optional[np.argsort(-scores[optional], kind="stable")]
    selected[optional[: target - int(selected.sum())]] = True
    return selected


def oof_single_swap_curve() -> dict[str, dict[str, int]]:
    with np.load(DATA, allow_pickle=False) as z:
        ptr = z["train_alarm_ptr"]
        labels = z["train_labels"].astype(bool)
        base = exact_count_mask(z["train_v11"], ptr, 3169)
    scores = np.load(EXP / "v30_meta_stack/station_extra_trees_oof.npy")
    actions = []
    for start, stop in zip(ptr[:-1], ptr[1:]):
        start, stop = int(start), int(stop)
        current = base[start:stop]
        k = int(current.sum())
        ranked = np.argsort(-scores[start:stop], kind="stable")[:k]
        proposed = np.zeros(stop - start, dtype=bool)
        proposed[ranked] = True
        added, removed = proposed & ~current, current & ~proposed
        if int(added.sum()) != 1 or int(removed.sum()) != 1:
            continue
        utility = float(scores[start:stop][added].sum() - scores[start:stop][removed].sum())
        delta = int((proposed & labels[start:stop]).sum() - (current & labels[start:stop]).sum())
        actions.append((utility, delta))
    actions.sort(reverse=True)
    curve = {}
    for size in (1, 3, 5, 8, 10, 15):
        subset = actions[:size]
        curve[str(size)] = {
            "net_delta_tp": int(sum(x[1] for x in subset)),
            "positive": int(sum(x[1] > 0 for x in subset)),
            "zero": int(sum(x[1] == 0 for x in subset)),
            "negative": int(sum(x[1] < 0 for x in subset)),
        }
    return curve


def main() -> None:
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    order_ids, base = load(BASE)
    _, full = load(FULL)
    with gzip.open(RECORDS, "rt", encoding="utf-8") as f:
        records = json.load(f)["test"]
    with np.load(EXP / "v25_semantic_router/cloud_dataset/v25_semantic_router.npz") as z:
        ptr = z["test_alarm_ptr"]
    scores = np.load(SCORES)
    score = {}
    for i, rec in enumerate(records):
        for j, alarm in enumerate(rec["alarms"]):
            score[(rec["order_id"], alarm["rid"])] = float(scores[int(ptr[i]) + j])
    proposals = []
    for oid in order_ids:
        old, new = ids(base[oid]), ids(full[oid])
        removed, added = old - new, new - old
        if len(removed) != 1 or len(added) != 1:
            continue
        gain = sum(score.get((oid, rid), 0.0) for rid in added) - sum(
            score.get((oid, rid), 0.0) for rid in removed
        )
        proposals.append({"order_id": oid, "removed": sorted(removed),
                          "added": sorted(added), "score_gain": gain})
    proposals.sort(key=lambda x: (x["score_gain"], x["order_id"]), reverse=True)
    chosen = proposals[:k]
    output = dict(base)
    for item in chosen:
        output[item["order_id"]] = full[item["order_id"]]
    total = sum(len(json.loads(raw)["rootcause"]) for raw in output.values())
    if total != 1035:
        raise RuntimeError(f"P changed to {total}")
    stem = f"v131_station_single_swap_top{k}_from_pair015"
    path = out_dir / f"{stem}.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["order_id", "output"])
        writer.writeheader()
        for oid in order_ids:
            writer.writerow({"order_id": oid, "output": output[oid]})
    meta = {"version": "v131-station-single-swap", "base": str(BASE),
            "base_sha256": sha256(BASE), "source_full_rerank": str(FULL),
            "source_scores": str(SCORES), "slice_size": k,
            "available_single_swaps": len(proposals), "selected": chosen,
            "oof_single_swap_curve": oof_single_swap_curve(),
            "predictions": total, "orders": len(order_ids), "sha256": sha256(path),
            "status": "exploratory_not_scored",
            "warning": "Model-derived only; no online score yet."}
    (out_dir / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
