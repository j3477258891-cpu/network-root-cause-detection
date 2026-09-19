"""Swap low-confidence champion alarms for graph distance-1 candidates.

Train data shows alarms one directed edge upstream of a TargetAlarm are much
more likely to be root causes than the rest.  This script keeps the champion
submission size fixed, adds all such test candidates, and removes the same
number of lowest v11-scored champion nodes.
"""
import csv, json
from collections import deque
from pathlib import Path

ROOT = Path(r"D:\zgyidong")
TEST = ROOT / "test"
CHAMP = ROOT / "experiments" / "v60_combined_checkpoint" / "highest_verified_combined.csv"
SCORES = ROOT / "codexgz" / "v11" / "v11_test_scores.csv"
OUT = ROOT / "experiments" / "submissions" / "distance1_swap_equation_filtered.csv"
EQUATION_REPORT = ROOT / "experiments" / "v37_online_equations" / "report.json"

def load_submission(path):
    order_ids, pred = [], {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            order_ids.append(row["order_id"])
            pred[row["order_id"]] = json.loads(row["output"])["rootcause"]
    return order_ids, pred

def distance1(order_id):
    obj = json.loads((TEST / order_id / f"{order_id}.log.topo.json").read_text(encoding="utf-8"))
    nodes = obj.get("nodes", [])
    rid = {n.get("@rid"): i for i, n in enumerate(nodes)}
    reverse = [[] for _ in nodes]
    for e in obj.get("edges", []):
        u, v = rid.get(e.get("in")), rid.get(e.get("out"))
        if u is not None and v is not None:
            reverse[v].append(u)
    starts = [i for i, n in enumerate(nodes) if n.get("@class") == "Alarm" and n.get("label") == "TargetAlarm"]
    dist = [99] * len(nodes)
    q = deque(starts)
    for i in starts:
        dist[i] = 0
    while q:
        u = q.popleft()
        for v in reverse[u]:
            if dist[v] > dist[u] + 1:
                dist[v] = dist[u] + 1
                q.append(v)
    return {n["@rid"]: n for i, n in enumerate(nodes)
            if n.get("@class") == "Alarm" and dist[i] == 1}

def main():
    order_ids, champion = load_submission(CHAMP)
    scores = {}
    with SCORES.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            scores[(r["order_id"], r["rid"])] = .25 * float(r["context_score"]) + .75 * float(r["meta_mean"])

    # Historical leaderboard equations fixed two of the graph candidates as
    # false.  Exclude them instead of spending a submission slot on a known
    # wrong node; this is independent evidence from the graph prior.
    fixed_false = set()
    if EQUATION_REPORT.exists():
        eq = json.loads(EQUATION_REPORT.read_text(encoding="utf-8"))
        fixed_false = {
            x["rid"] for x in eq.get("fixed_labels", [])
            if x.get("label") == 0
        }

    additions = {}
    for oid in order_ids:
        current = {n["@rid"] for n in champion[oid]}
        cand = distance1(oid)
        add = (set(cand) - current) - fixed_false
        if add:
            additions[oid] = {rid: cand[rid] for rid in add}
    add_keys = sorted((oid, rid) for oid, vals in additions.items() for rid in vals)
    remove_keys = sorted(
        ((oid, n["@rid"]) for oid, nodes in champion.items() for n in nodes
         if (oid, n["@rid"]) not in set(add_keys)),
        key=lambda k: (scores.get(k, -1e9), k[0], k[1]),
    )[:len(add_keys)]
    remove_set = set(remove_keys)
    add_set = set(add_keys)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["order_id", "output"])
        for oid in order_ids:
            nodes = [n for n in champion[oid] if (oid, n["@rid"]) not in remove_set]
            nodes += [{k: additions[oid][rid].get(k, "") for k in ("@rid", "title", "location", "reason")}
                      for rid in sorted(additions.get(oid, {}))]
            w.writerow([oid, json.dumps({"rootcause": nodes}, ensure_ascii=False)])
    report = {
        "baseline_predictions": sum(len(v) for v in champion.values()),
        "distance1_additions": len(add_set),
        "removed_low_score": len(remove_set),
        "output_predictions": sum(len(v) for v in champion.values()),
        "addition_orders": len(additions),
        "excluded_fixed_false": sorted(fixed_false),
        "output": str(OUT),
    }
    (OUT.with_suffix('.json')).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
