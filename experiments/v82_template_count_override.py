"""Use stable train-template root counts to correct current champion counts."""
from __future__ import annotations
import csv, gzip, json, hashlib, sys
from collections import Counter, defaultdict
from pathlib import Path
ROOT=Path(r"D:\zgyidong"); EXP=ROOT/"experiments"
sys.path.insert(0,str(EXP/"v30_meta_stack"))
from build_cross_order_probes import write_submission

RECORDS=EXP/"v25_semantic_router/cloud_dataset/semantic_records.json.gz"
BASE=EXP/"v60_combined_checkpoint/highest_verified_combined.csv"
OUT=EXP/"v82_template_count_override"
TRUE_ROOTS=1044

def load_base():
    d={}
    with BASE.open("r",encoding="utf-8-sig",newline="") as h:
        for row in csv.DictReader(h): d[row["order_id"]]={x["@rid"] for x in json.loads(row["output"])["rootcause"]}
    return d

def main():
    with gzip.open(RECORDS,"rt",encoding="utf-8") as h: data=json.load(h)
    train,test=data["train"],data["test"]
    groups=defaultdict(list)
    for o in train: groups[repr(o["signature"])].append(o)
    known={}; evidence={}
    for o in test:
        refs=groups.get(repr(o["signature"]),[])
        ks=[sum(int(a["is_root"]) for a in r["alarms"]) for r in refs]
        if len(ks)<2: continue
        k,n=Counter(ks).most_common(1)[0]
        if n/len(ks)>=.9:
            known[o["order_id"]]=int(k);evidence[o["order_id"]]={"support":len(ks),"k":int(k),"confidence":n/len(ks)}
    base=load_base(); changed=[]; roots={};
    # Current champion is retained as the ranking source.  For a known count,
    # retain the highest-scoring current roots and add the strongest omitted
    # nodes according to V30 consensus test scores.
    import numpy as np
    scores=np.load(EXP/"v30_meta_stack/v30_consensus_test.npy")
    score_by={}
    pos=0
    for o in test:
        for a in o["alarms"]: score_by[(o["order_id"],a["rid"])]=float(scores[pos]);pos+=1
    for o in test:
        oid=o["order_id"]; current=set(base[oid]); target=known.get(oid,len(current));
        ranked=sorted((a["rid"] for a in o["alarms"]),key=lambda rid:(-score_by[(oid,rid)],rid))
        if oid not in known: roots[oid]=current;continue
        selected=set(current)
        if len(selected)>target:
            selected=set(sorted(selected,key=lambda rid:(-score_by[(oid,rid)],rid))[:target])
        elif len(selected)<target:
            for rid in ranked:
                if rid not in selected:
                    selected.add(rid)
                    if len(selected)==target:break
        roots[oid]=selected
        if selected!=current: changed.append({"order_id":oid,"before":len(current),"after":len(selected),"remove":sorted(current-selected),"add":sorted(selected-current),"evidence":evidence[oid]})
    # Build output using source alarm fields, preserving all other orders.
    records_by={o["order_id"]:o for o in test}; out_roots={}
    for o in test:
        selected=roots[o["order_id"]]; out_roots[o["order_id"]]=[next(a for a in o["alarms"] if a["rid"]==rid) for rid in selected]
    OUT.mkdir(parents=True,exist_ok=True); path=OUT/"result_record_v82_template_count.csv";write_submission(path,[o["order_id"] for o in test],out_roots)
    report={"version":"v82-template-count-override-1","base":str(BASE),"known_orders":len(known),"known_count_distribution":dict(Counter(known.values())),"changed_orders":len(changed),"total_before":sum(len(v) for v in base.values()),"total_after":sum(len(v) for v in roots.values()),"delta_predictions":sum(len(v) for v in roots.values())-sum(len(v) for v in base.values()),"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"changed":changed,"warning":"Uses stable template count priors; public score not yet verified."}
    (OUT/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf8");print(json.dumps({k:v for k,v in report.items() if k!='changed'},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
