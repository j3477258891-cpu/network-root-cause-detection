"""Combine disjoint V126 high-margin swaps with V127 V30 cross-order pairs."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(r"D:\zgyidong")
OUT = ROOT / "experiments/v127_v30_campaign"
BASE = ROOT / "experiments/v124_campaign/v124_single_pair_015_from_probe13.csv"
V126 = ROOT / "experiments/v126_v16_campaign/v126_v16_highmargin_top4_from_pair015.csv"
V127 = OUT / "v127_v30_cross_current_top5_from_pair015.csv"


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


def roots(row: dict) -> set[str]:
    return {x["@rid"] for x in json.loads(row["output"])["rootcause"]}


def main() -> None:
    base, v126, v127 = load(BASE), load(V126), load(V127)
    if set(base) != set(v126) or set(base) != set(v127):
        raise RuntimeError("order sets differ")
    changed126 = {o for o in base if roots(base[o]) != roots(v126[o])}
    changed127 = {o for o in base if roots(base[o]) != roots(v127[o])}
    overlap = changed126 & changed127
    if overlap:
        raise RuntimeError(f"action order overlap: {sorted(overlap)}")
    out = dict(v127)
    for o in changed126:
        out[o] = v126[o]
    total = sum(len(json.loads(r["output"])["rootcause"]) for r in out.values())
    if total != 1035:
        raise RuntimeError(f"prediction count changed: {total}")
    path = OUT / "v127_combo_v126top4_v30top5_from_pair015.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["order_id", "output"])
        w.writeheader()
        for o in sorted(out):
            w.writerow(out[o])
    meta = {
        "version": "v127-v126-v30-disjoint-combo",
        "base": str(BASE), "base_sha256": sha256(BASE),
        "v126_source": str(V126), "v126_sha256": sha256(V126),
        "v127_source": str(V127), "v127_sha256": sha256(V127),
        "base_public_f1": 0.921597, "base_tp": 958,
        "v126_action_orders": len(changed126), "v127_action_orders": len(changed127),
        "changed_orders": len(changed126 | changed127), "predictions": total,
        "orders": len(out), "sha256": sha256(path),
        "status": "exploratory_not_scored",
        "target_condition": "if all 9 component groups net +1, TP=967, F1=0.930452",
        "warning": "Two independent model routes are combined; no public score evidence yet.",
    }
    (OUT / "v127_combo_v126top4_v30top5_from_pair015.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
