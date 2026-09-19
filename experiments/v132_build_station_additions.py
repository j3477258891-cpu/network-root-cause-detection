"""Build pure-addition probes from the station ExtraTrees ranking.

Unlike a swap, an addition never removes a currently selected root.  At fixed
baseline TP=958/P=1035, one true addition raises F1; this route is therefore a
useful independent recall-expansion test after the V124 swap probes.
"""
from __future__ import annotations

import csv, gzip, hashlib, json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(r"D:\zgyidong")
EXP = ROOT / "experiments"
BASE = EXP / "v124_campaign/v124_single_pair_015_from_probe13.csv"
DATA = EXP / "v25_semantic_router/cloud_dataset/v25_semantic_router.npz"
RECORDS = EXP / "v25_semantic_router/cloud_dataset/semantic_records.json.gz"
SCORES = EXP / "v30_meta_stack/station_extra_trees_test.npy"
V37 = EXP / "v37_online_equations/report.json"
OUT = EXP / "v132_station_additions"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_base():
    order_ids, roots = [], {}
    with BASE.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            order_ids.append(row["order_id"])
            roots[row["order_id"]] = json.loads(row["output"])["rootcause"]
    return order_ids, roots


def fixed_false():
    blocked = set()
    if V37.exists():
        report = json.loads(V37.read_text(encoding="utf-8"))
        blocked |= {(x["order_id"], x["rid"]) for x in report.get("fixed_labels", [])
                    if x.get("label") == 0}
    # Directly observed online delta=-1 actions imply add=0 and remove=1.
    # Do not reintroduce these roots through the pure-addition route.
    blocked |= {
        ("4079fac3-5a5c-48e1-b171-d09393a78fc9", "#-1:def8e14e-d628-48d0-bec0-a1cdc239c1b3"),
        ("faeeeb88-468e-485d-b71b-d3f799baaaf0", "#-1:9602714c-0fcb-43e5-b630-e6036059b9c5"),
        ("70b84b0d-e9d5-4b40-b8dd-aef24ac338b3", "#-1:c63b9fec-3656-43fc-adb8-e114e98247d4"),
        ("c27a096b-576b-4fb7-9fb2-7f04feab066d", "#-1:bdb85be4-661a-4a6a-9f56-fbff53833e80"),
        ("808e05f4-3151-4437-9845-4994bc189ed0", "#-1:4cf9e7ef-7cf0-4d22-b79a-77787a7ceb02"),
    }
    return blocked


def oof_addition_curve():
    """Historical station-OOF precision of one top addition per order."""
    with np.load(DATA, allow_pickle=False) as z:
        ptr, labels, v11 = z["train_alarm_ptr"], z["train_labels"].astype(bool), z["train_v11"]
    selected = np.zeros(len(v11), dtype=bool)
    optional = []
    for st, sp in zip(ptr[:-1], ptr[1:]):
        st, sp = int(st), int(sp)
        rr = np.argsort(-v11[st:sp], kind="stable")[:8]
        selected[st + rr[0]] = True
        optional.extend((st + rr[1:]).tolist())
    optional = np.asarray(optional, dtype=np.int64)
    optional = optional[np.argsort(-v11[optional], kind="stable")]
    selected[optional[:3169 - int(selected.sum())]] = True
    scores = np.load(EXP / "v30_meta_stack/station_extra_trees_oof.npy")
    order_of = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
    counts = np.add.reduceat(selected.astype(int), ptr[:-1])
    rows = [int(i) for i in np.argsort(-scores, kind="stable")
            if not selected[i] and counts[order_of[i]] < 8]
    seen, chosen = set(), []
    for i in rows:
        oi = int(order_of[i])
        if oi not in seen:
            seen.add(oi); chosen.append(i)
    out = {}
    for n in (1, 3, 5, 10, 20, 30):
        a = chosen[:n]
        out[str(n)] = {"true": int(labels[a].sum()), "false": int(n - labels[a].sum()),
                       "precision": round(float(labels[a].mean()), 4)}
    return out


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    order_ids, base = load_base()
    with gzip.open(RECORDS, "rt", encoding="utf-8") as f:
        records = json.load(f)["test"]
    with np.load(DATA, allow_pickle=False) as z:
        ptr = z["test_alarm_ptr"]
    scores = np.load(SCORES)
    blocked = fixed_false()
    proposals = []
    for i, rec in enumerate(records):
        oid = rec["order_id"]
        current = {x["@rid"] for x in base[oid]}
        if len(current) >= 8:
            continue
        ranked = sorted(
            ((float(scores[int(ptr[i]) + j]), alarm) for j, alarm in enumerate(rec["alarms"])
             if alarm["rid"] not in current and (oid, alarm["rid"]) not in blocked),
            key=lambda x: (x[0], x[1]["rid"]), reverse=True,
        )
        if ranked:
            score, alarm = ranked[0]
            proposals.append({"order_id": oid, "add_rid": alarm["rid"],
                              "station_score": score, "base_root_count": len(current)})
    proposals.sort(key=lambda x: (x["station_score"], x["order_id"]), reverse=True)
    chosen = proposals[:k]
    output = {oid: list(base[oid]) for oid in order_ids}
    by_oid = {r["order_id"]: r for r in records}
    for item in chosen:
        alarm = next(a for a in by_oid[item["order_id"]]["alarms"] if a["rid"] == item["add_rid"])
        source = alarm.get("source", {})
        output[item["order_id"]].append({"@rid": alarm["rid"], "title": source.get("title", ""),
                                          "location": source.get("location", ""), "reason": source.get("reason", "")})
    total = sum(len(nodes) for nodes in output.values())
    if total != 1035 + k:
        raise RuntimeError(f"P changed to {total}, expected {1035+k}")
    stem = f"v132_station_add_top{k}_from_pair015"
    path = out_dir / f"{stem}.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["order_id", "output"]); writer.writeheader()
        for oid in order_ids:
            writer.writerow({"order_id": oid, "output": json.dumps({"rootcause": output[oid]}, ensure_ascii=False)})
    meta = {"version": "v132-station-pure-addition", "base": str(BASE),
            "base_sha256": sha256(BASE), "source_scores": str(SCORES),
            "slice_size": k, "available_additions": len(proposals), "selected": chosen,
            "oof_station_addition_curve": oof_addition_curve(),
            "predictions": total, "orders": len(order_ids), "sha256": sha256(path),
            "conditional_scores": {"all_selected_additions_true": round(2 * (958 + k) / (2079 + k), 9),
                                   "none_selected_additions_true": round(2 * 958 / (2079 + k), 9)},
            "status": "exploratory_not_scored",
            "warning": "Pure additions are model-derived; no online score yet."}
    (out_dir / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
