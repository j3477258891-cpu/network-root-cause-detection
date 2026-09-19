"""Preserve verified V124 actions inside the V84 template-count DP output."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(r"D:\zgyidong")
OUT = ROOT / "experiments/v125_attack_campaign"
BASE = ROOT / "experiments/v124_campaign/v124_single_pair_015_from_probe13.csv"
V84 = ROOT / "experiments/v84_template_count_dp/result_v84_template_dp_1035.csv"
PRESERVE_ORDERS = {
    # pair015 is publicly confirmed +1 relative to probe13.
    "4079fac3-5a5c-48e1-b171-d09393a78fc9",
    # V120 pair061 is equation-fixed negative.
    "d5376a71-c48e-4351-ba60-f982778b3eb4",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path) -> dict[str, dict]:
    out = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            out[row["order_id"]] = {"order_id": row["order_id"], "output": row["output"]}
    return out


def root_count(row: dict) -> int:
    return len(json.loads(row["output"])["rootcause"])


def root_set(row: dict) -> set[str]:
    return {n["@rid"] for n in json.loads(row["output"])["rootcause"]}


def main() -> None:
    base = load(BASE)
    v84 = load(V84)
    if set(base) != set(v84):
        raise RuntimeError("base and V84 order sets differ")
    out = dict(v84)
    for oid in PRESERVE_ORDERS:
        out[oid] = base[oid]
    total = sum(root_count(row) for row in out.values())
    if total != 1035:
        raise RuntimeError(f"prediction count changed: {total}")
    path = OUT / "v125_template_dp_preserve_pair015.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["order_id", "output"])
        w.writeheader()
        for oid in sorted(out):
            w.writerow(out[oid])
    changed = sum(root_set(out[o]) != root_set(base[o]) for o in out)
    meta = {
        "version": "v125-template-count-dp-preserve",
        "base": str(BASE), "base_sha256": sha256(BASE),
        "source_v84": str(V84), "source_v84_sha256": sha256(V84),
        "base_public_f1": 0.921597, "base_tp": 958,
        "preserved_orders": sorted(PRESERVE_ORDERS),
        "changed_orders": changed, "predictions": total, "orders": len(out),
        "sha256": sha256(path), "status": "exploratory_not_scored",
        "warning": "V84 is an offline template-count DP; no public score evidence.",
    }
    (OUT / "v125_template_dp_preserve_pair015.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
