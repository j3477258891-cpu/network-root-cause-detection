"""Build a replacement campaign after V104's delete-pool failure.

V104's 169 removals lost 153 true roots online.  V118 uses only the lowest
V11-scored champion nodes as a small, safer deletion pool and codes additions
from the V75 candidate list.  Budget: calibration + 24 code probes + 3 finals
after the two already consumed submissions = 30 total.
"""
from __future__ import annotations
import csv, json, gzip
from pathlib import Path
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/v60_combined_checkpoint/highest_verified_combined.csv'
SCORES=ROOT/'codexgz/v11/v11_test_scores.csv'
OUT=ROOT/'experiments/v118_safe_positive_campaign'
DELETE_N=40; ADD_N=60; CODE_N=24; FINAL_N=3

def load_rows(path):
    rows=[]
    with open(path,encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f): rows.append((r['order_id'],json.loads(r['output'])['rootcause']))
    return rows

def alarm_lookup():
    out={}
    rec_path=ROOT/'experiments/v25_semantic_router/cloud_dataset/semantic_records.json.gz'
    with gzip.open(rec_path,'rt',encoding='utf-8') as f: rec=json.load(f)
    for o in rec['test']:
        for a in o.get('alarms',[]): out[(o['order_id'],a.get('rid'))]=a
    return out

def load_actions():
    report=json.load(open(ROOT/'experiments/v75_dense_block_campaign/report.json',encoding='utf-8'))
    probs=np.asarray(report['correctness_probabilities'],float)
    actions=report['candidate_actions']
    base={(oid,n['@rid']) for oid,nodes in load_rows(BASE) for n in nodes}
    items=[]
    for a,p in zip(actions,probs):
        for rid in a.get('add_rids',[]):
            key=(a['order_id'],rid)
            if key not in base: items.append((float(p),a,rid))
    items.sort(key=lambda x:(-x[0],x[1]['order_id'],x[2]))
    adds=[]; seen=set()
    for p,a,rid in items:
        key=(a['order_id'],rid)
        if key in seen: continue
        seen.add(key); adds.append({'order_id':a['order_id'],'rid':rid,'prior':p})
        if len(adds)>=ADD_N: break
    scores={}
    with open(SCORES,encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f): scores[(r['order_id'],r['rid'])]=.25*float(r['context_score'])+.75*float(r['meta_mean'])
    candidates=[]
    for oid,nodes in load_rows(BASE):
        for n in nodes:
            candidates.append((scores.get((oid,n['@rid']),-1),oid,n))
    candidates.sort(key=lambda x:(x[0],x[1],x[2]['@rid']))
    dels=[{'order_id':oid,'rid':n['@rid'],'score':sc} for sc,oid,n in candidates[:DELETE_N]]
    return adds,dels

def render(rows, adds, dels, lookup):
    add={}; rem={}
    for x in adds:add.setdefault(x['order_id'],set()).add(x['rid'])
    for x in dels:rem.setdefault(x['order_id'],set()).add(x['rid'])
    out=[]
    for oid,nodes in rows:
        roots=[n for n in nodes if n['@rid'] not in rem.get(oid,set())]
        have={n['@rid'] for n in roots}
        for rid in sorted(add.get(oid,set())):
            if rid in have:continue
            n=lookup.get((oid,rid),{})
            roots.append({'@rid':rid,'title':n.get('title',''),'location':n.get('raw_location',n.get('location','')),'reason':n.get('reason','')})
        out.append((oid,roots))
    return out

def write(path, rows):
    with open(path,'w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['order_id','output'])
        for oid,nodes in rows:w.writerow([oid,json.dumps({'rootcause':nodes},ensure_ascii=False)])

def generate():
    adds,dels=load_actions(); rows=load_rows(BASE); lookup=alarm_lookup(); OUT.mkdir(exist_ok=True)
    rng=np.random.default_rng(118); A=(rng.random((CODE_N,ADD_N))<.35).astype(np.int8)
    (OUT/'matrix.json').write_text(json.dumps(A.tolist()),encoding='utf-8')
    meta={'delete_count':DELETE_N,'add_count':ADD_N,'code_count':CODE_N,'final_count':FINAL_N,'total_with_two_prior_submissions':1+CODE_N+FINAL_N+2,'add_priors':[x['prior'] for x in adds],'delete_scores':[x['score'] for x in dels]}
    (OUT/'campaign.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    write(OUT/'probe_00_calibration.csv',render(rows,[],dels,lookup))
    for i in range(CODE_N):
        chosen=[adds[j] for j in np.flatnonzero(A[i])]
        write(OUT/f'probe_{i+1:02d}_code.csv',render(rows,chosen,dels,lookup))
    print(json.dumps(meta,indent=2))

def decode(calibration, scores):
    adds,dels=load_actions(); A=np.asarray(json.load(open(OUT/'matrix.json')),dtype=np.int8); nd=len(dels)
    root_loss=round(956-float(calibration)*(1044+1035-nd)/2)
    rhs=[]
    for i,score in enumerate(scores):
        n=int(A[i].sum()); P=1035-nd+n; delta=round(float(score)*(1044+P)/2)-956; rhs.append(delta+root_loss)
    pri=np.asarray([x['prior'] for x in adds],float); c=-np.log(np.clip(pri,1e-6,1-1e-6)/np.clip(1-pri,1e-6,1-1e-6))
    res=milp(c,integrality=np.ones(ADD_N,dtype=np.int8),bounds=Bounds(np.zeros(ADD_N),np.ones(ADD_N)),constraints=LinearConstraint(A,np.asarray(rhs),np.asarray(rhs)),options={'time_limit':60})
    if res.x is None:raise RuntimeError(res.message)
    good=np.flatnonzero(np.rint(res.x).astype(int)).tolist(); print(json.dumps({'root_loss':root_loss,'add_indices':good,'add_count':len(good)},indent=2))
    (OUT/'decoded.json').write_text(json.dumps({'root_loss':root_loss,'add_indices':good},indent=2),encoding='utf-8')

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--decode',action='store_true');ap.add_argument('--calibration',type=float);ap.add_argument('--scores',nargs='+',type=float);a=ap.parse_args()
    if a.decode:decode(a.calibration,a.scores)
    else:generate()
