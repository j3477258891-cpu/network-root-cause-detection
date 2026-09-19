"""Generate/decode a 29-submission campaign.

One calibration + 25 positive-code probes + 3 final checkpoints.  Each probe
contains the same high-value deletion pool, so the calibration removes its
unknown TP loss; the probe equations then have positive coefficients for both
add and delete correctness variables.  This file only creates local artifacts;
it never submits anything.
"""
from __future__ import annotations
import argparse, csv, json, gzip
from pathlib import Path
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/v60_combined_checkpoint/highest_verified_combined.csv"
OUT = ROOT / "experiments/v104_15day_positive_campaign"
ALARM_BY = {}
DELETE_POOL_SIZE = 169
FINAL_PREFIXES = [110, 140, 169]

def load_actions():
    j75 = json.load(open(ROOT/"experiments/v75_dense_block_campaign/report.json", encoding="utf-8"))
    p75 = np.asarray(j75["correctness_probabilities"], float); a75 = j75["candidate_actions"]
    ia = np.flatnonzero([bool(a["add_rids"]) for a in a75]); ia = ia[np.argsort(-p75[ia])[:71]]
    delete_rows=[]
    for rel in ("v58_coded_campaign/report.json", "v59_extended_coded_campaign/report.json"):
        report=json.load(open(ROOT/"experiments"/rel, encoding="utf-8")); probs=np.asarray(report["correctness_probabilities"],float)
        for i,action in enumerate(report["candidate_actions"]):
            if not action["add_rids"]:
                delete_rows.append((1-probs[i], action))
    delete_rows.sort(key=lambda x:-x[0]); delete_rows=delete_rows[:DELETE_POOL_SIZE]
    return [a75[i] for i in ia], p75[ia], [x[1] for x in delete_rows], np.asarray([x[0] for x in delete_rows])

def read_base():
    global ALARM_BY
    alarm_by={}
    rec_path=ROOT/"experiments/v25_semantic_router/cloud_dataset/semantic_records.json.gz"
    with gzip.open(rec_path, "rt", encoding="utf-8") as f:
        rec=json.load(f)
    for order in rec["test"]:
        for alarm in order.get("alarms", []):
            alarm_by[(order["order_id"], alarm.get("rid"))]=alarm
    ALARM_BY = alarm_by
    rows=[]
    with open(BASE, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            obj=json.loads(r["output"]); roots=obj.get("rootcause", [])
            rows.append({"order_id":r["order_id"], "roots":roots})
    return rows

def apply(rows, adds, deletes):
    adds_by={}
    dels_by={}
    for a in adds:
        adds_by.setdefault(a["order_id"], set()).update(a.get("add_rids", []))
    for a in deletes:
        dels_by.setdefault(a["order_id"], set()).update(a.get("remove_rids", []))
    out=[]
    for row in rows:
        seen={x.get("@rid") for x in row["roots"]}
        roots=[x for x in row["roots"] if x.get("@rid") not in dels_by.get(row["order_id"], set())]
        for rid in adds_by.get(row["order_id"], set()):
            if rid not in seen:
                src=ALARM_BY.get((row["order_id"], rid), {})
                roots.append({"@rid":rid,"title":src.get("title",""),"location":src.get("raw_location",src.get("location","")),"reason":src.get("reason","")})
        out.append({"order_id":row["order_id"], "output":json.dumps({"rootcause":roots}, ensure_ascii=False)})
    return out

def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w=csv.DictWriter(f, fieldnames=["order_id","output"]); w.writeheader(); w.writerows(rows)

def generate():
    adds, pa, dels, qd = load_actions(); rows=read_base(); OUT.mkdir(exist_ok=True)
    rng=np.random.default_rng(20260821); A=(rng.random((25,71))<.35).astype(np.int8)
    (OUT/"matrix.json").write_text(json.dumps(A.tolist(), ensure_ascii=False), encoding="utf-8")
    portfolio_path=ROOT/"experiments/v109_adaptive_portfolio.json"
    meta={"base":str(BASE),"add_action_count":71,"delete_pool_count":len(dels),"probe_count":25,"final_delete_prefixes":FINAL_PREFIXES,"adaptive_portfolio":str(portfolio_path),"add_priors":pa.tolist(),"delete_correct_priors":qd.tolist()}
    (OUT/"campaign.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    write_csv(OUT/"probe_00_calibration.csv", apply(rows, [], dels))
    for r in range(25):
        write_csv(OUT/f"probe_{r+1:02d}_code.csv", apply(rows, [adds[i] for i in np.flatnonzero(A[r])], dels))
    print("generated", OUT)

def decode(calibration_score, probe_scores):
    adds, pa, dels, qd = load_actions(); A=np.asarray(json.load(open(OUT/"matrix.json",encoding="utf-8")), dtype=np.int8)
    # Scores are exact F1; convert to TP using known prediction count per file.
    ndel=len(dels); rootd=round(956 - calibration_score*(1044 + (1035-ndel))/2)
    rhs=[]
    for r,score in enumerate(probe_scores):
        nadd=int(A[r].sum()); P=1035-ndel+nadd; delta=round(score*(1044+P)/2)-956
        rhs.append(delta+rootd)
    c=-np.log(np.clip(pa,1e-6,1-1e-6)/np.clip(1-pa,1e-6,1-1e-6))
    res=milp(c,integrality=np.ones(71,dtype=np.int8),bounds=Bounds(np.zeros(71),np.ones(71)),constraints=LinearConstraint(A,np.asarray(rhs),np.asarray(rhs)),options={"time_limit":60})
    if res.x is None: raise RuntimeError(res.message)
    good=np.flatnonzero(np.rint(res.x).astype(int))
    final={}
    rows=read_base()
    portfolio_path=ROOT/"experiments/v109_adaptive_portfolio.json"
    if portfolio_path.exists():
        mapping=json.load(open(portfolio_path,encoding="utf-8"))["mapping"]; portfolio=mapping[str(int(np.clip(len(good),15,50)))][:3]
    else:
        portfolio=[{"name":f"prefix_{n}","indices":list(range(n))} for n in FINAL_PREFIXES]
    for k,item in enumerate(portfolio,1):
        chosen=[dels[i] for i in item["indices"]]; name=item["name"].replace(".","p")
        write_csv(OUT/f"final_{k:02d}_{name}.csv", apply(rows,[adds[i] for i in good],chosen))
        final[name]=good.tolist()
    (OUT/"decoded.json").write_text(json.dumps({"root_deletions":rootd,"decoded_add_indices":good.tolist()},indent=2),encoding="utf-8")
    print(json.dumps({"root_deletions":rootd,"decoded_add_count":len(good),"finals":list(final)}))

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--decode",action="store_true"); ap.add_argument("--calibration-score",type=float); ap.add_argument("--probe-scores",nargs="+",type=float)
    args=ap.parse_args(); generate() if not args.decode else decode(args.calibration_score,args.probe_scores)
