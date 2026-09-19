import json
from pathlib import Path
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
ROOT=Path(__file__).resolve().parents[1]
def main():
 A=np.array(json.load(open(ROOT/'experiments/v104_15day_positive_campaign/matrix.json'))); j=json.load(open(ROOT/'experiments/v75_dense_block_campaign/report.json',encoding='utf-8')); p=np.array(j['correctness_probabilities']); a=j['candidate_actions']; ids=np.flatnonzero([bool(x['add_rids']) for x in a]); ids=ids[np.argsort(-p[ids])[:71]]; p=p[ids]; c=-np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1-1e-6)); rng=np.random.default_rng(3)
 for t in range(5):
  y=(rng.random(71)<p).astype(np.int8); cons=[LinearConstraint(A,A@y,A@y)]; sols=[]
  for k in range(8):
   r=milp(c,integrality=np.ones(71,dtype=np.int8),bounds=Bounds(np.zeros(71),np.ones(71)),constraints=cons,options={'time_limit':20});
   if r.x is None: break
   z=np.rint(r.x).astype(np.int8); sols.append(z); coef=np.where(z==1,-1,1); lb=1-int(z.sum()); cons.append(LinearConstraint(coef[None,:],lb,np.inf))
  print(t,'solutions',len(sols),'true_rank',next((i for i,z in enumerate(sols) if np.array_equal(z,y)),None),'best_acc',max((z==y).mean() for z in sols))
if __name__=='__main__':main()
