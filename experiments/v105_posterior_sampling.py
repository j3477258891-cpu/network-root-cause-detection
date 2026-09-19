import json
from pathlib import Path
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
ROOT=Path(__file__).resolve().parents[1]
def main():
 j=json.load(open(ROOT/'experiments/v75_dense_block_campaign/report.json',encoding='utf-8')); p75=np.array(j['correctness_probabilities']); a=j['candidate_actions']; ia=np.flatnonzero([bool(x['add_rids']) for x in a]); ia=ia[np.argsort(-p75[ia])[:71]]
 j=json.load(open(ROOT/'experiments/v59_extended_coded_campaign/report.json',encoding='utf-8')); p59=np.array(j['correctness_probabilities']); a=j['candidate_actions']; id=np.flatnonzero([not bool(x['add_rids']) for x in a]); id=id[np.argsort(-(1-p59[id]))[:90]]
 p=np.r_[p75[ia],1-p59[id]]; n=len(p); rng=np.random.default_rng(30); A=(rng.random((29,n))<.12).astype(np.int8); base=-np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1-1e-6))
 for trial in range(2):
  y=(rng.random(n)<p).astype(int); rhs=A@y; sols=[]
  for k in range(30):
   c=base+rng.normal(0,1.0,n); r=milp(c,integrality=np.ones(n,dtype=np.int8),bounds=Bounds(np.zeros(n),np.ones(n)),constraints=LinearConstraint(A,rhs,rhs),options={'time_limit':8});
   if r.x is not None: sols.append(np.rint(r.x).astype(int))
  Z=np.mean(sols,axis=0); print('trial',trial,'samples',len(sols),'freqtrue',float(((Z>0.5)==y).mean()))
  best=(-1,None)
  for ta in np.arange(.1,1.01,.1):
   for td in np.arange(.1,1.01,.1):
    sel=np.r_[(Z[:71]>=ta),(Z[71:]>=td)]; dt=int((sel[:71]*y[:71]).sum())-int((sel[71:]*(1-y[71:])).sum()); dp=int(sel[:71].sum())-int(sel[71:].sum()); f=2*(956+dt)/(2079+dp)
    if f>best[0]: best=(f,(ta,td,int(sel[:71].sum()),int(sel[71:].sum())))
  print('oracle threshold best',best)
if __name__=='__main__':main()
