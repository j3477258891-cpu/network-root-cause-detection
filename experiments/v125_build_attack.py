"""Build high-risk V125 attack probes from the verified V124 champion.

This does not upload anything.  It creates P=1035 files that retain the
verified pair015 improvement and apply one or more previously untested V120
swaps.  The route is intentionally exploratory: model priors are not online
evidence and every output must be scored before being treated as useful.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(r"D:\zgyidong")
EXP = ROOT / "experiments"
OUT = EXP / "v125_attack_campaign"
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))
import v124_build_group_probes as v124  # noqa: E402

BASE = ROOT / "experiments/v124_campaign/v124_single_pair_015_from_probe13.csv"
CATALOG = ROOT / "experiments/v120_swap_campaign/candidate_catalog.json"

TESTED = {
    "v120_pair_008", "v120_pair_014", "v120_pair_044",
    "v120_pair_049", "v120_pair_084", "v120_pair_015",
    "v120_pair_011", "v120_pair_018", "v120_pair_045",
    "v120_pair_030", "v120_pair_054", "v120_pair_033",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({"order_id": row["order_id"],
                         "roots": json.loads(row["output"])["rootcause"]})
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["order_id", "output"])
        w.writeheader()
        for row in rows:
            output = row.get("output")
            if output is None:
                output = json.dumps({"rootcause": row["roots"]}, ensure_ascii=False)
            w.writerow({"order_id": row["order_id"], "output": output})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default="v120_pair_038")
    ap.add_argument("--top-n", type=int, default=0,
                    help="if >0, build one file with top-N untested unique-order swaps")
    args = ap.parse_args()

    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))["candidates"]
    by_id = {c["candidate_id"]: c for c in catalog}
    alarms, _ = v124.v121.load_alarm_records()
    rows = load_rows(BASE)
    base_by_order = {r["order_id"]: {n["@rid"] for n in r["roots"]} for r in rows}

    if args.top_n > 0:
        chosen = []
        seen_orders = set()
        for c in sorted(catalog, key=lambda x: (x.get("model_score", -9),
                                                 x.get("nominal_p_net_gain", 0)), reverse=True):
            if c["candidate_id"] in TESTED or c["order_id"] in seen_orders:
                continue
            if c.get("kind", "swap") != "swap":
                continue
            present = base_by_order.get(c["order_id"], set())
            if c["remove_rid"] not in present or c["add_rid"] in present:
                continue
            chosen.append(c)
            seen_orders.add(c["order_id"])
            if len(chosen) >= args.top_n:
                break
    else:
        chosen = [by_id[args.candidate]]

    actions = []
    for c in chosen:
        present = base_by_order.get(c["order_id"], set())
        if c["remove_rid"] not in present:
            raise RuntimeError(f"remove RID not in base: {c['candidate_id']}")
        if c["add_rid"] in present:
            raise RuntimeError(f"add RID already in base: {c['candidate_id']}")
        actions.append({"order_id": c["order_id"],
                        "remove_rid": c["remove_rid"],
                        "add_rid": c["add_rid"]})

    out_rows = v124.apply_actions(rows, actions, alarms)
    valid, predictions, message = v124.validate(out_rows, 1035)
    if not valid:
        raise RuntimeError(message)
    suffix = args.candidate if args.top_n <= 0 else f"top{args.top_n}"
    path = OUT / f"v125_{suffix}_from_pair015.csv"
    write_csv(path, out_rows)
    metadata = {
        "version": "v125-high-risk-attack",
        "base": str(BASE),
        "base_sha256": sha256(BASE),
        "base_public_f1": 0.921597,
        "base_tp": 958,
        "candidate_ids": [c["candidate_id"] for c in chosen],
        "actions": actions,
        "model_scores": [c.get("model_score") for c in chosen],
        "nominal_p_net_gain": [c.get("nominal_p_net_gain") for c in chosen],
        "predictions": predictions,
        "orders": len(out_rows),
        "sha256": sha256(path),
        "status": "exploratory_not_scored",
        "warning": "No model prior is online evidence; do not call this a verified gain before scoring.",
    }
    (OUT / f"v125_{suffix}_from_pair015.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
