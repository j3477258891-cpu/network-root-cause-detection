import json
from pathlib import Path
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
ROOT=Path(__file__).resolve().parents[1]
def solve(A,rhs,p):
 c=-np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1-1e-6)); r=milp(c,integrality=np.ones(len(p),dtype=np.int8),bounds=Bounds(np.zeros(len(p)),np.ones(len(p))),constraints=LinearConstraint(A,rhs,rhs),options={'time_limit':10}); return None if r.x is None else np.rint(r.x).astype(np.int8)
def main():
 j75=json.load(open(ROOT/'experiments/v75_dense_block_campaign/report.json',encoding='utf-8')); p75=np.array(j75['correctness_probabilities']); a75=j75['candidate_actions']; ia=np.flatnonzero([bool(a['add_rids']) for a in a75]); ia=ia[np.argsort(-p75[ia])]; ia=ia[:71]; pa=p75[ia]
 j59=json.load(open(ROOT/'experiments/v59_extended_coded_campaign/report.json',encoding='utf-8')); p59=np.array(j59['correctness_probabilities']); a59=j59['candidate_actions']; id=np.flatnonzero([not bool(a['add_rids']) for a in a59]); id=id[np.argsort(-(1-p59[id]))][:70]; pd=1-p59[id]
 rng=np.random.default_rng(20260821); A=(rng.random((28,71))<.35).astype(np.int8); fs=[]; exact=[]
 for t in range(10):
  yd=(rng.random(70)<pd).astype(np.int8); ya=(rng.random(71)<pa).astype(np.int8); rootd=70-yd.sum(); z=solve(A,A@ya,pa)
  if z is None: continue
  exact.append(int(np.array_equal(z,ya))); addsel=int((z&ya).sum()); # decoded z=1 gets added
  # all 70 deletions are submitted; root deletions hurt TP, nonroots reduce P
  tp=956+addsel-(70-yd.sum()); P=1035+int(z.sum())-70
  fs.append(2*tp/(1044+P))
 print({'mean_f1':float(np.mean(fs)),'p945':float(np.mean(np.array(fs)>=.945)),'exact_add':float(np.mean(exact)),'q_delete':float(pd.mean()),'n':len(fs),'quantiles':np.quantile(fs,[.05,.5,.95]).tolist()})
if __name__=='__main__':main()
