import json,time
from pathlib import Path
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
ROOT=Path(__file__).resolve().parents[1]
def main():
 j=json.load(open(ROOT/'experiments/v75_dense_block_campaign/report.json',encoding='utf-8')); p=np.array(j['correctness_probabilities']); a=j['candidate_actions']; ids=np.flatnonzero([bool(x['add_rids']) for x in a]); ids=ids[np.argsort(-p[ids])[:71]]; p=p[ids]; c=-np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1-1e-6)); rng=np.random.default_rng(115); ys=[(rng.random(71)<p).astype(np.int8) for _ in range(5)]; best=(-1,-1,None)
 for density in (.18,.25,.32,.4,.5):
  for seed in range(6):
   A=(np.random.default_rng(seed+int(density*1000)).random((25,71))<density).astype(np.int8); ex=acc=0
   for y in ys:
    r=milp(c,integrality=np.ones(71,dtype=np.int8),bounds=Bounds(np.zeros(71),np.ones(71)),constraints=LinearConstraint(A,A@y,A@y),options={'time_limit':5});
    if r.x is None:continue
    z=np.rint(r.x).astype(np.int8); ex+=int(np.array_equal(z,y)); acc+=(z==y).mean()
   score=(ex/5,acc/5)
   if score>best[:2]: best=(score[0],score[1],A); print({'density':density,'seed':seed,'exact':score[0],'accuracy':score[1]},flush=True)
 np.save(ROOT/'experiments/v115_best_matrix.npy',best[2]); print('BEST',best[:2])
if __name__=='__main__':main()
