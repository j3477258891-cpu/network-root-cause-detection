"""Combine disjoint V126 high-margin swaps with V129 station rerank slice."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(r"D:\zgyidong")
EXP = ROOT / "experiments"
OUT = EXP / "v130_combo"
BASE = EXP / "v124_campaign/v124_single_pair_015_from_probe13.csv"
V126 = EXP / "v126_v16_campaign/v126_v16_highmargin_top4_from_pair015.csv"
V129 = EXP / "v129_station_rerank/v129_station_samecount_top5_from_pair015.csv"


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


def ids(raw):
    return {x["@rid"] for x in json.loads(raw)["rootcause"]}


def main() -> None:
    order_ids, base = load(BASE)
    _, v126 = load(V126)
    _, v129 = load(V129)
    c126 = {o for o in order_ids if ids(base[o]) != ids(v126[o])}
    c129 = {o for o in order_ids if ids(base[o]) != ids(v129[o])}
    if c126 & c129:
        raise RuntimeError(f"overlap: {sorted(c126 & c129)}")
    out = dict(v129)
    for o in c126:
        out[o] = v126[o]
    total = sum(len(json.loads(raw)["rootcause"]) for raw in out.values())
    if total != 1035:
        raise RuntimeError(f"P changed to {total}")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "v130_combo_v126top4_v129stationtop5_from_pair015.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["order_id", "output"])
        w.writeheader()
        for o in order_ids:
            w.writerow({"order_id": o, "output": out[o]})
    meta = {
        "version": "v130-v126-v129-combo", "base": str(BASE),
        "base_sha256": sha256(BASE), "v126": str(V126), "v126_sha256": sha256(V126),
        "v129": str(V129), "v129_sha256": sha256(V129),
        "base_public_f1": 0.921597, "base_tp": 958,
        "v126_changed_orders": len(c126), "v129_changed_orders": len(c129),
        "changed_orders": len(c126 | c129), "predictions": total,
        "orders": len(order_ids), "sha256": sha256(path),
        "status": "exploratory_not_scored",
        "target_condition": "if all 9 component groups net +1, TP=967, F1=0.930452",
        "warning": "OOF/model evidence only; no public score yet.",
    }
    (OUT / "v130_combo_v126top4_v129stationtop5_from_pair015.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
