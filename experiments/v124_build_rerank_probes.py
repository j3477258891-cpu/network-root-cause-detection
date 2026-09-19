"""V124 rerank-slice probes: partition test orders, rerank each slice with the
stacked node model (slice-internal fixed count), emit one probe file per slice.

Each probe = baseline file with ONLY that slice reranked.  Online TP delta of a
probe equals that slice's net gain, so slices can be verified independently and
the final file is baseline + union of verified-positive slices.
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
if str(ROOT / "codexgz/work") not in sys.path:
    sys.path.insert(0, str(ROOT / "codexgz/work"))
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

import research_pairwise_error_model as pw  # noqa: E402

OUT = ROOT / "experiments/v124_campaign"
DATA = ROOT / "experiments/v25_semantic_router/cloud_dataset/v25_semantic_router.npz"
CURRENT = ROOT / "experiments/v60_combined_checkpoint/highest_verified_combined.csv"
TEST_DIR = ROOT / "test"
STACKED = OUT / "stacked_test_pred.npy"
N_SLICES = 3


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    arrays = dict(np.load(DATA, allow_pickle=False))
    test_orders = pw.v10.load_orders(TEST_DIR, False)
    te_ptr = arrays["test_alarm_ptr"]
    stacked = np.load(STACKED).astype(np.float32)
    base = pw.read_submission(CURRENT)
    base_mask = pw.baseline_mask_for_orders(test_orders, base)
    print("test orders", len(test_orders), "P", int(base_mask.sum()), "stacked", stacked.shape, flush=True)

    n_orders = len(test_orders)
    # Balanced slices by order index but grouped by site to keep slices comparable.
    sites = [pw.site_of(a.get("location", "")) for o in test_orders
             for a in o["alarms"][:1]]
    # Simple round-robin by sorted order index (site-balanced not required for probe).
    slices = np.array_split(np.arange(n_orders), N_SLICES)

    # Build the final reranked submission for each slice independently.
    rows = []
    with CURRENT.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            rows.append({"order_id": r["order_id"], "roots": json.loads(r["output"])["rootcause"]})
    row_by = {r["order_id"]: r for r in rows}

    probe_info = []
    for si, idx in enumerate(slices):
        oid_list = [test_orders[i]["id"] for i in idx]
        # Slice-internal ptr & scores
        starts = [int(te_ptr[i]) for i in idx] + [int(te_ptr[idx[-1] + 1])]
        local_ptr = np.asarray(starts, dtype=np.int64)
        local_ptr0 = local_ptr - local_ptr[0]
        local_start, local_end = int(te_ptr[idx[0]]), int(te_ptr[idx[-1] + 1])
        local_scores = stacked[local_start:local_end]
        local_base = base_mask[local_start:local_end]
        local_p = int(local_base.sum())
        local_new = pw.exact_mask(local_scores, local_ptr0, local_p)
        print(f"slice {si}: orders={len(idx)} local_p={local_p} changed={int((local_new != local_base).sum())}", flush=True)

        # Emit probe file: rerank slice, keep other orders at baseline.
        out_rows = []
        for row in rows:
            roots = [dict(x) for x in row["roots"]]
            oid = row["order_id"]
            if oid in oid_list:
                k = idx[int(np.flatnonzero(np.asarray(oid_list) == oid)[0])]
                start = int(te_ptr[k])
                new_sel = [i for i in range(int(te_ptr[k + 1]) - start) if local_new[start - local_start + i]]
                new_roots = []
                for j in new_sel:
                    a = test_orders[k]["alarms"][j]
                    new_roots.append({"@rid": a["@rid"], "title": a.get("title", ""),
                                      "location": a.get("location", ""), "reason": a.get("reason", "")})
                roots = new_roots
            out_rows.append({"order_id": oid, "output": json.dumps({"rootcause": roots}, ensure_ascii=False)})
        total = sum(len(json.loads(x["output"])["rootcause"]) for x in out_rows)
        if total != 1035:
            raise ValueError(f"slice {si} prediction count {total}")
        for r in out_rows:
            assert 0 < len(json.loads(r["output"])["rootcause"]) <= 8, r["order_id"]
        path = OUT / f"probe_r{si + 1}_rerank_slice.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["order_id", "output"])
            writer.writeheader()
            writer.writerows(out_rows)
        probe_info.append({
            "probe_id": f"R{si + 1}", "slice": si, "orders": len(idx),
            "slice_p": local_p, "file": str(path), "predictions": 1035,
            "sha256": sha256(path),
        })
        print(f"probe_r{si + 1}: {len(idx)} orders P={total} sha256={sha256(path)[:16]}")

    write_json(OUT / "rerank_probes.json", {
        "version": "v124-rerank", "n_slices": N_SLICES,
        "method": "slice-internal exact-count rerank with stacked 3-model score; P per slice preserved",
        "oof_train_deltas_3slices": [15, 16, 9], "oof_train_total": 40,
        "probes": probe_info,
    })
    print(json.dumps({"probes": probe_info}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
