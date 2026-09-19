import json
from pathlib import Path
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
ROOT=Path(__file__).resolve().parents[1]
def solve(A,rhs,p):
 c=-np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1-1e-6)); r=milp(c,integrality=np.ones(len(p),dtype=np.int8),bounds=Bounds(np.zeros(len(p)),np.ones(len(p))),constraints=LinearConstraint(A,rhs,rhs),options={'time_limit':10}); return None if r.x is None else np.rint(r.x).astype(np.int8)
def main():
 j=json.load(open(ROOT/'experiments/v59_extended_coded_campaign/report.json',encoding='utf-8')); p0=np.array(j['correctness_probabilities']); acts=j['candidate_actions']; ids=np.flatnonzero([not bool(a['add_rids']) for a in acts]); q=1-p0[ids]; ids=ids[np.argsort(-q)];
 pa=np.array([p0[i] for i,a in enumerate(acts) if bool(a['add_rids'])]); addids=np.flatnonzero([bool(a['add_rids']) for a in acts]); addids=addids[np.argsort(-p0[addids])[:3]]; addgain=float(p0[addids].sum())
 rng=np.random.default_rng(20260821); fs=[]; exact=[]
 for t in range(100):
  blocks=[]; ok=True; d=0
  for m in (15,14):
   bi=ids[len(blocks)*35:(len(blocks)+1)*35]; p=1-p0[bi]; A=(rng.random((m,35))<.35).astype(np.int8); y=(rng.random(35)<p).astype(np.int8); z=solve(A,A@y,p)
   if z is None: ok=False; break
   exact.append(int(np.array_equal(z,y))); d += int((z*y).sum())
  if ok:
   # add top 3 are blindly included; expected proxy uses sampled labels not expectation
   ya=(rng.random(3)<p0[addids]).astype(np.int8); d_tp=int(ya.sum())
   # for each delete block 70 actions, applied decoded z; errors mean action count not simply d.
   # approximate exact-only score; retain exact flag.
   f=2*(956+d_tp)/(2079+3- d)
   fs.append(f)
 print({'mean_f1_proxy':float(np.mean(fs)),'p945':float(np.mean(np.array(fs)>=.945)),'block_exact_rate':float(np.mean(exact)),'n':len(fs)})
if __name__=='__main__':main()
