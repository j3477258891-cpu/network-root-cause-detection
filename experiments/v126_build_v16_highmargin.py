"""Build V126 swaps from the highest calibrated V16 OOF margins."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(r"D:\zgyidong")
EXP = ROOT / "experiments"
OUT = EXP / "v126_v16_campaign"
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))
import v124_build_group_probes as v124  # noqa: E402

BASE = ROOT / "experiments/v124_campaign/v124_single_pair_015_from_probe13.csv"
CATALOG = ROOT / "experiments/v16/v16_test_swap_catalog.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            rows.append({"order_id": r["order_id"],
                         "roots": json.loads(r["output"])["rootcause"]})
    return rows


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows(BASE)
    present = {r["order_id"]: {z["@rid"] for z in r["roots"]} for r in rows}
    candidates = []
    with CATALOG.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["removed_rid"] not in present.get(r["order_id"], set()):
                continue
            if r["added_rid"] in present.get(r["order_id"], set()):
                continue
            candidates.append(r)
    # One action per order, sorted by seed-minimum margin.  This uses V16's
    # held-out calibration: margin >= .6 had 6/10 positive labels in OOF and
    # the four largest test margins are the strongest available candidates.
    chosen = []
    seen_orders = set()
    for r in sorted(candidates, key=lambda x: float(x["seed_min_margin"]), reverse=True):
        if r["order_id"] in seen_orders:
            continue
        chosen.append(r)
        seen_orders.add(r["order_id"])
        if len(chosen) >= n:
            break
    if len(chosen) < n:
        raise RuntimeError(f"only {len(chosen)} valid V16 candidates")
    alarms, _ = v124.v121.load_alarm_records()
    actions = [{"order_id": r["order_id"], "remove_rid": r["removed_rid"], "add_rid": r["added_rid"]}
               for r in chosen]
    out = v124.apply_actions(rows, actions, alarms)
    valid, predictions, message = v124.validate(out, 1035)
    if not valid:
        raise RuntimeError(message)
    stem = f"v126_v16_highmargin_top{n}_from_pair015"
    path = OUT / f"{stem}.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["order_id", "output"])
        w.writeheader()
        for r in out:
            w.writerow(r)
    meta = {
        "version": "v126-v16-highmargin",
        "base": str(BASE), "base_sha256": sha256(BASE),
        "base_public_f1": 0.921597, "base_tp": 958,
        "source_catalog": str(CATALOG),
        "oof_calibration": {"margin_ge_0.7": {"positive": 4, "total": 5},
                             "margin_ge_0.6": {"positive": 6, "total": 10}},
        "candidates": [{k: r[k] for k in ["order_id", "removed_rid", "added_rid", "margin", "seed_min_margin", "high_conflict", "meta_mean"]}
                       for r in chosen],
        "predictions": predictions, "orders": len(out), "sha256": sha256(path),
        "status": "exploratory_not_scored",
        "warning": "V16 OOF calibration and model margins are not online evidence.",
        "target_condition": f"if all {n} swaps net +1, TP={958+n}, F1={2*(958+n)/2079:.6f}",
    }
    (OUT / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
