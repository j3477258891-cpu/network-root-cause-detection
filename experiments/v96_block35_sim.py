import json
from pathlib import Path
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
ROOT=Path(__file__).resolve().parents[1]
def solve(A,rhs,p):
 c=-np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1-1e-6)); r=milp(c,integrality=np.ones(len(p),dtype=np.int8),bounds=Bounds(np.zeros(len(p)),np.ones(len(p))),constraints=LinearConstraint(A,rhs,rhs),options={'time_limit':10}); return None if r.x is None else np.rint(r.x).astype(np.int8)
def main():
 j=json.load(open(ROOT/'experiments/v59_extended_coded_campaign/report.json',encoding='utf-8')); p0=np.array(j['correctness_probabilities']); acts=j['candidate_actions']; ids=np.flatnonzero([not bool(a['add_rids']) for a in acts]); q=1-p0[ids]; ids=ids[np.argsort(-q)]; p=1-p0[ids[:35]]; rng=np.random.default_rng(12)
 for m in (14,15,16):
  vals=[]
  for t in range(20):
   A=(rng.random((m,35))<.35).astype(np.int8); y=(rng.random(35)<p).astype(np.int8); z=solve(A,A@y,p); vals.append(None if z is None else int(np.array_equal(z,y)))
  print(m,np.mean([x for x in vals if x is not None]),sum(x is None for x in vals))
if __name__=='__main__':main()
