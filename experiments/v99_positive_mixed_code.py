import json,time
from pathlib import Path
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
ROOT=Path(__file__).resolve().parents[1]
def solve(A,rhs,p,limit=30):
 c=-np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1-1e-6)); r=milp(c,integrality=np.ones(len(p),dtype=np.int8),bounds=Bounds(np.zeros(len(p)),np.ones(len(p))),constraints=LinearConstraint(A,rhs,rhs),options={'time_limit':limit,'mip_rel_gap':0}); return None if r.x is None else np.rint(r.x).astype(np.int8)
def main():
 j75=json.load(open(ROOT/'experiments/v75_dense_block_campaign/report.json',encoding='utf-8')); p75=np.array(j75['correctness_probabilities']); a75=j75['candidate_actions']; ia=np.flatnonzero([bool(a['add_rids']) for a in a75]); ia=ia[np.argsort(-p75[ia])][:71]
 j59=json.load(open(ROOT/'experiments/v59_extended_coded_campaign/report.json',encoding='utf-8')); p59=np.array(j59['correctness_probabilities']); a59=j59['candidate_actions']; id=np.flatnonzero([not bool(a['add_rids']) for a in a59]); id=id[np.argsort(-(1-p59[id]))]
 for nd in (50,90):
  p=np.r_[p75[ia],1-p59[id[:nd]]]; n=len(p); rng=np.random.default_rng(20260821); out=[]
  for density in (.12,.15):
   A=(rng.random((29,n))<density).astype(np.int8); ex=acc=skip=0; vals=[]
   for t in range(20):
    y=(rng.random(n)<p).astype(np.int8); z=solve(A,A@y,p)
    if z is None: skip+=1; continue
    ex+=int(np.array_equal(z,y)); acc+=int((z==y).sum()); app=z; ya=y[:71]; za=app[:71]; zd=app[71:]
    yd=y[71:]
    d_tp=int((za*ya).sum())-int((zd*(1-yd)).sum()); d_p=int(za.sum())-int(zd.sum()); vals.append(2*(956+d_tp)/(2079+d_p))
   out.append({'nd':nd,'density':density,'n':n,'exact':ex/max(1,20-skip),'accuracy':acc/max(1,(20-skip)*n),'skip':skip,'mean_f1':float(np.mean(vals)) if vals else None,'p945':float(np.mean(np.array(vals)>=.945)) if vals else None})
  print(out,flush=True)
if __name__=='__main__':main()
