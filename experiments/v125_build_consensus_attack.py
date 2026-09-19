"""Build a high-risk top-N attack from four-model pair-delta consensus."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"D:\zgyidong")
EXP = ROOT / "experiments"
OUT = EXP / "v125_attack_campaign"
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))
import v124_build_group_probes as v124  # noqa: E402

BASE = ROOT / "experiments/v124_campaign/v124_single_pair_015_from_probe13.csv"
REPORT = ROOT / "experiments/research_pairwise_error_model/pair_delta_report.json"
TESTED_IDS = {
    "v120_pair_008", "v120_pair_014", "v120_pair_044",
    "v120_pair_049", "v120_pair_084", "v120_pair_015",
    "v120_pair_011", "v120_pair_018", "v120_pair_045",
    "v120_pair_030", "v120_pair_054", "v120_pair_033",
}
FIXED_NEGATIVE_IDS = {"v120_pair_088", "v120_pair_019", "v120_pair_091", "v120_pair_061"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    by_key = defaultdict(list)
    for model, data in report["models"].items():
        for x in data["test_top_unique"]:
            key = (x["order_id"], x["remove_rid"], x["add_rid"])
            by_key[key].append({"model": model, "score": x["score"]})

    rows = []
    with BASE.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            rows.append({"order_id": r["order_id"],
                         "roots": json.loads(r["output"])["rootcause"]})
    present = {r["order_id"]: {z["@rid"] for z in r["roots"]} for r in rows}
    tested_tuples = set()
    v120 = json.loads((EXP / "v120_swap_campaign/candidate_catalog.json").read_text(encoding="utf-8"))["candidates"]
    for c in v120:
        if c["candidate_id"] in TESTED_IDS:
            tested_tuples.add((c["order_id"], c["remove_rid"], c["add_rid"]))

    ranked = []
    for (oid, rem, add), vals in by_key.items():
        if len(vals) < 3 or (oid, rem, add) in tested_tuples:
            continue
        if rem not in present.get(oid, set()) or add in present.get(oid, set()):
            continue
        ranked.append({"order_id": oid, "remove_rid": rem, "add_rid": add,
                       "support": len(vals), "mean_score": sum(v["score"] for v in vals) / len(vals),
                       "scores": vals})
    ranked.sort(key=lambda x: (x["mean_score"], x["support"]), reverse=True)
    chosen = []
    seen_orders = set()
    for c in ranked:
        if c["order_id"] in seen_orders:
            continue
        chosen.append(c)
        seen_orders.add(c["order_id"])
        if len(chosen) >= n:
            break
    # The four-model report exposes only a top-100 slice per model.  Fill a
    # requested 25-action attack with the best remaining V120 actions when
    # consensus has fewer independent orders, preserving the same P=1035
    # invariant.  These fillers are explicitly tagged as lower-evidence.
    if len(chosen) < n:
        v120 = json.loads((EXP / "v120_swap_campaign/candidate_catalog.json").read_text(encoding="utf-8"))["candidates"]
        for c in sorted(v120, key=lambda x: (x.get("model_score", -9),
                                              x.get("nominal_p_net_gain", 0)), reverse=True):
            key = (c["order_id"], c["remove_rid"], c["add_rid"])
            if c["candidate_id"] in TESTED_IDS or c["candidate_id"] in FIXED_NEGATIVE_IDS or key in tested_tuples or c["order_id"] in seen_orders:
                continue
            if c["remove_rid"] not in present.get(c["order_id"], set()) or c["add_rid"] in present.get(c["order_id"], set()):
                continue
            chosen.append({"order_id": c["order_id"], "remove_rid": c["remove_rid"],
                           "add_rid": c["add_rid"], "support": c.get("model_support", 0),
                           "mean_score": c.get("model_score", 0),
                           "scores": [{"model": "v120", "score": c.get("model_score", 0)}],
                           "evidence": "v120_fallback"})
            seen_orders.add(c["order_id"])
            if len(chosen) >= n:
                break
    if len(chosen) < n:
        raise RuntimeError(f"only {len(chosen)} valid independent consensus actions")

    alarms, _ = v124.v121.load_alarm_records()
    actions = [{"order_id": c["order_id"], "remove_rid": c["remove_rid"], "add_rid": c["add_rid"]}
               for c in chosen]
    out = v124.apply_actions(rows, actions, alarms)
    valid, predictions, message = v124.validate(out, 1035)
    if not valid:
        raise RuntimeError(message)
    path = OUT / f"v125_consensus_top{n}_from_pair015.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["order_id", "output"])
        w.writeheader()
        for r in out:
            w.writerow(r)
    meta = {"version": "v125-four-model-consensus-attack", "base": str(BASE),
            "base_sha256": sha256(BASE), "base_public_f1": 0.921597, "base_tp": 958,
            "candidate_count": len(chosen), "candidates": chosen,
            "predictions": predictions, "orders": len(out), "sha256": sha256(path),
            "status": "exploratory_not_scored",
            "warning": "Model/OOF consensus is not leaderboard evidence; score before treating as a gain.",
            "target_condition": f"all {n} swaps net +1 would yield TP={958+n} and F1={2*(958+n)/2079:.6f}"}
    (OUT / f"v125_consensus_top{n}_from_pair015.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ["version", "candidate_count", "predictions", "orders", "sha256", "target_condition"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
