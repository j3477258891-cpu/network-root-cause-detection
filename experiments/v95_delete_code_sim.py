import json,time
from pathlib import Path
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
ROOT=Path(__file__).resolve().parents[1]
def solve(A,rhs,p):
 c=-np.log(np.clip(p,1e-5,1-1e-5)/np.clip(1-p,1e-5,1-1e-5))
 r=milp(c,integrality=np.ones(len(p),dtype=np.int8),bounds=Bounds(np.zeros(len(p)),np.ones(len(p))),constraints=LinearConstraint(A,rhs,rhs),options={'time_limit':30,'mip_rel_gap':0})
 return None if r.x is None else np.rint(r.x).astype(np.int8)
def main():
 j=json.load(open(ROOT/'experiments/v59_extended_coded_campaign/report.json',encoding='utf-8')); p0=np.asarray(j['correctness_probabilities'],float); acts=j['candidate_actions']; delete=np.array([not bool(a['add_rids']) for a in acts]); ids=np.flatnonzero(delete); q=1-p0[ids]; order=ids[np.argsort(-q)]; ids=order[:70]; p=1-p0[ids]; rng=np.random.default_rng(20260821); out=[]
 for dens in (.15,.25,.35):
  A=(rng.random((29,len(p)))<dens).astype(np.int8); ex=acc=skip=0; fs=[]
  for t in range(5):
   y=(rng.random(len(p))<p).astype(np.int8); z=solve(A,A@y,p)
   if z is None: skip+=1; continue
   ex+=int(np.array_equal(z,y)); acc+=int((z==y).sum()); d=int((z*y).sum()); fs.append(2*956/(2079-d))
  row={'density':dens,'exact':ex/max(1,10-skip),'accuracy':acc/max(1,(10-skip)*len(p)),'skip':skip,'mean_f1':float(np.mean(fs)) if fs else None,'p945':float(np.mean(np.array(fs)>=.945)) if fs else None}; print(row,flush=True); out.append(row)
 (ROOT/'experiments/v95_delete_code_sim.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
if __name__=='__main__':main()
