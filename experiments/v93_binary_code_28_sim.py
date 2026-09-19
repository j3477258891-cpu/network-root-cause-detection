import json
from pathlib import Path
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
ROOT=Path(__file__).resolve().parents[1]
def solve(A,rhs,p):
 n=len(p); c=-np.log(np.maximum(p,1e-6)/np.maximum(1-p,1e-6)); r=milp(c,integrality=np.ones(n,dtype=np.int8),bounds=Bounds(np.zeros(n),np.ones(n)),constraints=LinearConstraint(A,rhs,rhs),options={'time_limit':3.0}); return None if r.x is None else np.rint(r.x).astype(np.int8)
def main():
 j1=json.load(open(ROOT/'experiments/v58_coded_campaign/report.json',encoding='utf-8')); j2=json.load(open(ROOT/'experiments/v59_extended_coded_campaign/report.json',encoding='utf-8')); p0=np.asarray(j1['correctness_probabilities']+j2['correctness_probabilities']); add=np.asarray([bool(x['add_rids']) for x in j1['candidate_actions']+j2['candidate_actions']]); ia=np.flatnonzero(add)[np.argsort(-p0[add])][:71]; id=np.flatnonzero(~add)[np.argsort(-p0[~add])][:19]; p=np.r_[p0[ia],p0[id]]; rng=np.random.default_rng(5)
 for density in (.3,):
  n=len(p); A=(rng.random((29,n))<density).astype(np.int8); ex=acc=sk=rt=0
  for _ in range(5):
   y=(rng.random(n)<p).astype(np.int8); z=solve(A,A@y,p)
   if z is None:sk+=1;continue
   ex+=int(np.array_equal(z,y)); acc+=int((z==y).sum()); rt+=int((z&y).sum())
  print({'density':density,'exact':ex/5,'acc':acc/(5*n),'recover_true':rt/5,'skip':sk})
if __name__=='__main__':main()
