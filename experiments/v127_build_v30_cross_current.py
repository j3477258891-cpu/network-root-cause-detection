"""Rebuild V30 consensus cross-order swaps on the current V124 champion.

V30's original probes used an obsolete V29 champion.  This rebuild applies
the same score model to the current champion, excludes previously tested
orders, and preserves P=1035 by pairing one add with one delete per pair.
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
OUT = EXP / "v127_v30_campaign"
if str(EXP / "v30_meta_stack") not in sys.path:
    sys.path.insert(0, str(EXP / "v30_meta_stack"))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))
import build_cross_order_probes as v30  # noqa: E402

BASE = EXP / "v124_campaign/v124_single_pair_015_from_probe13.csv"
SCORES = EXP / "v30_meta_stack/v30_consensus_test.npy"
RECORDS = EXP / "v25_semantic_router/cloud_dataset/semantic_records.json.gz"
TRUE_ROOTS = 1044
PREDICTIONS = 1035

# Do not reuse orders already used in V124/V126 probes.  The V30 model is
# being tested as an independent source, not mixed with known actions.
EXCLUDED_ORDERS = {
    "4079fac3-5a5c-48e1-b171-d09393a78fc9",
    "06d84a90-670f-4ccb-b482-9afdb874d164",
    "aa321b94-f3d9-45cf-b1ee-f3a1fe38da68",
    "b92c1c42-bc69-48fa-b5c6-7bbd2635a15b",
    "faeeeb88-468e-485d-b71b-d3f799baaaf0",
    "12fb9fb0-19b0-4b39-b037-f2e479fadb8a",
    "51d2ff1d-1bcb-41c7-ae33-83db088b2545",
    "ab02ed9d-74ea-4fc1-8935-9e5396cf94ba",
    "70b84b0d-e9d5-4b40-b8dd-aef24ac338b3",
    "c27a096b-576b-4fb7-9fb2-7f04feab066d",
    "808e05f4-3151-4437-9845-4994bc189ed0",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_base(path: Path):
    order_ids, roots = [], {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            order_ids.append(r["order_id"])
            roots[r["order_id"]] = json.loads(r["output"])["rootcause"]
    return order_ids, roots


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(EXP / "v25_semantic_router/cloud_dataset/v25_semantic_router.npz") as z:
        ptr = z["test_alarm_ptr"]
    with gzip.open(RECORDS, "rt", encoding="utf-8") as f:
        records = json.load(f)["test"]
    order_ids, roots = load_base(BASE)
    records_by_order = {r["order_id"]: r for r in records}
    scores = np.load(SCORES)
    pairs, row_alarm = v30.build_pairs(records, ptr, scores, roots, EXCLUDED_ORDERS)
    if len(pairs) < n:
        raise RuntimeError(f"only {len(pairs)} current-base pairs")
    subset = pairs[:n]
    actions = v30.actions_for_pairs(subset, records, row_alarm)
    out_roots = v30.apply_actions(roots, records_by_order, actions)
    predictions = sum(len(x) for x in out_roots.values())
    if predictions != PREDICTIONS:
        raise RuntimeError(f"P changed to {predictions}")
    stem = f"v127_v30_cross_current_top{n}_from_pair015"
    path = OUT / f"{stem}.csv"
    v30.write_submission(path, order_ids, out_roots)
    meta = {
        "version": "v127-v30-cross-order-current-base",
        "base": str(BASE), "base_sha256": sha256(BASE),
        "base_public_f1": 0.921597, "base_tp": 958,
        "source_scores": str(SCORES), "excluded_orders": len(EXCLUDED_ORDERS),
        "available_pairs": len(pairs), "pair_count": n,
        "pairs": subset, "actions": actions,
        "predictions": predictions, "orders": len(order_ids),
        "sha256": sha256(path), "status": "exploratory_not_scored",
        "target_condition": f"all {n} pairs net +1 gives TP={958+n}, F1={2*(958+n)/2079:.6f}",
        "warning": "V30 OOF/public results used an older base; this current-base rebuild is unscored.",
    }
    (OUT / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ["version", "available_pairs", "pair_count", "predictions", "orders", "sha256", "target_condition"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
