import json
from pathlib import Path
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
ROOT=Path(__file__).resolve().parents[1]
def main():
 j=json.load(open(ROOT/'experiments/v75_dense_block_campaign/report.json',encoding='utf-8')); p75=np.array(j['correctness_probabilities']); a=j['candidate_actions']; ia=np.flatnonzero([bool(x['add_rids']) for x in a]); ia=ia[np.argsort(-p75[ia])[:71]]; j=json.load(open(ROOT/'experiments/v59_extended_coded_campaign/report.json',encoding='utf-8')); p59=np.array(j['correctness_probabilities']); a=j['candidate_actions']; id=np.flatnonzero([not bool(x['add_rids']) for x in a]); id=id[np.argsort(-(1-p59[id]))[:90]]; p=np.r_[p75[ia],1-p59[id]]; n=len(p); rng=np.random.default_rng(13); A=(rng.random((29,n))<.12).astype(np.int8); base=-np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1-1e-6))
 for t in range(3):
  y=(rng.random(n)<p).astype(int); rhs=A@y; zs=[]
  for k in range(12):
   c=base+rng.normal(0,.08,n); r=milp(c,integrality=np.ones(n,dtype=np.int8),bounds=Bounds(np.zeros(n),np.ones(n)),constraints=LinearConstraint(A,rhs,rhs),options={'time_limit':10}); zs.append(np.rint(r.x).astype(int))
  Z=np.mean(zs,axis=0)
  print('trial',t,'freq',[(th, round(float(np.mean((Z[:71]>=th)*y[:71])),3), round(float(np.mean((Z[71:]>=th)==y[71:])),3)) for th in [.3,.5,.7,.9]])
  for th in [.3,.5,.7,.9]:
   sel=(Z>=th).astype(int); dt=int((sel[:71]*y[:71]).sum())-int((sel[71:]*(1-y[71:])).sum()); dp=int(sel[:71].sum())-int(sel[71:].sum()); print(th,2*(956+dt)/(2079+dp))
if __name__=='__main__':main()
