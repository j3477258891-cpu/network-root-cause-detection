"""Build pure-addition probes ranked by the V30 consensus model."""
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
SCORES = EXP / "v30_meta_stack/v30_consensus_test.npy"
V37 = EXP / "v37_online_equations/report.json"
OUT = EXP / "v133_consensus_additions"


def sha256(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def load_base():
    ids, roots = [], {}
    with BASE.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            ids.append(r["order_id"]); roots[r["order_id"]] = json.loads(r["output"])["rootcause"]
    return ids, roots


def blocked_nodes():
    out = set()
    if V37.exists():
        d = json.loads(V37.read_text(encoding="utf-8"))
        out |= {(x["order_id"], x["rid"]) for x in d.get("fixed_labels", []) if x.get("label") == 0}
    # Online delta=-1 actions prove these additions false.
    out |= {
        ("4079fac3-5a5c-48e1-b171-d09393a78fc9", "#-1:def8e14e-d628-48d0-bec0-a1cdc239c1b3"),
        ("faeeeb88-468e-485d-b71b-d3f799baaaf0", "#-1:9602714c-0fcb-43e5-b630-e6036059b9c5"),
        ("70b84b0d-e9d5-4b40-b8dd-aef24ac338b3", "#-1:c63b9fec-3656-43fc-adb8-e114e98247d4"),
        ("c27a096b-576b-4fb7-9fb2-7f04feab066d", "#-1:bdb85be4-661a-4a6a-9f56-fbff53833e80"),
        ("808e05f4-3151-4437-9845-4994bc189ed0", "#-1:4cf9e7ef-7cf0-4d22-b79a-77787a7ceb02"),
    }
    return out


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    order_ids, base = load_base()
    with gzip.open(RECORDS, "rt", encoding="utf-8") as f: records = json.load(f)["test"]
    with np.load(DATA, allow_pickle=False) as z: ptr = z["test_alarm_ptr"]
    scores = np.load(SCORES); blocked = blocked_nodes(); proposals = []
    for i, rec in enumerate(records):
        oid = rec["order_id"]; current = {x["@rid"] for x in base[oid]}
        if len(current) >= 8: continue
        candidates = [(float(scores[int(ptr[i]) + j]), a) for j, a in enumerate(rec["alarms"])
                      if a["rid"] not in current and (oid, a["rid"]) not in blocked]
        if candidates:
            score, alarm = max(candidates, key=lambda x: (x[0], x[1]["rid"]))
            proposals.append({"order_id": oid, "add_rid": alarm["rid"], "consensus_score": score,
                              "base_root_count": len(current)})
    proposals.sort(key=lambda x: (x["consensus_score"], x["order_id"]), reverse=True)
    chosen = proposals[:k]; output = {oid: list(base[oid]) for oid in order_ids}
    rec_map = {r["order_id"]: r for r in records}
    for item in chosen:
        alarm = next(a for a in rec_map[item["order_id"]]["alarms"] if a["rid"] == item["add_rid"])
        source = alarm.get("source", {})
        output[item["order_id"]].append({"@rid": alarm["rid"], "title": source.get("title", ""),
                                          "location": source.get("location", ""), "reason": source.get("reason", "")})
    total = sum(len(nodes) for nodes in output.values())
    if total != 1035 + k: raise RuntimeError((total, k))
    stem = f"v133_consensus_add_top{k}_from_pair015"; path = out_dir / f"{stem}.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["order_id", "output"]); w.writeheader()
        for oid in order_ids: w.writerow({"order_id": oid, "output": json.dumps({"rootcause": output[oid]}, ensure_ascii=False)})
    meta = {"version": "v133-consensus-pure-addition", "base": str(BASE), "base_sha256": sha256(BASE),
            "source_scores": str(SCORES), "slice_size": k, "available_additions": len(proposals),
            "selected": chosen, "predictions": total, "orders": len(order_ids), "sha256": sha256(path),
            "status": "exploratory_not_scored", "warning": "Consensus ranking is model-derived; no online score yet."}
    (out_dir / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
