import json,time
from pathlib import Path
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
ROOT=Path(__file__).resolve().parents[1]
def main():
 A0=np.array(json.load(open(ROOT/'experiments/v104_15day_positive_campaign/matrix.json',encoding='utf-8')),dtype=np.int8)
 j=json.load(open(ROOT/'experiments/v75_dense_block_campaign/report.json',encoding='utf-8')); p=np.array(j['correctness_probabilities']); a=j['candidate_actions']; ids=np.flatnonzero([bool(x['add_rids']) for x in a]); ids=ids[np.argsort(-p[ids])[:71]]; p=p[ids]; c=-np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1-1e-6)); rng=np.random.default_rng(44); ys=[(rng.random(71)<p).astype(int) for _ in range(3)]; rows=[]
 for drop in range(25):
  A=np.delete(A0,drop,axis=0); exact=0; acc=0
  for y in ys:
   r=milp(c,integrality=np.ones(71,dtype=np.int8),bounds=Bounds(np.zeros(71),np.ones(71)),constraints=LinearConstraint(A,A@y,A@y),options={'time_limit':3}); z=None if r.x is None else np.rint(r.x).astype(int)
   if z is not None: exact+=int(np.array_equal(z,y)); acc+=(z==y).sum()
  row={'drop':drop,'exact':exact/3,'accuracy':acc/(3*71)}; print(row,flush=True); rows.append(row)
 best=max(rows,key=lambda x:(x['exact'],x['accuracy'])); json.dump({'best':best,'rows':rows},open(ROOT/'experiments/v106_drop_row_search.json','w'),indent=2); print('BEST',best)
if __name__=='__main__':main()
