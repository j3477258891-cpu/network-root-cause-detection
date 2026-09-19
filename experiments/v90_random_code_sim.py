from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
ROOT=Path(__file__).resolve().parents[1]

def solve(A,rhs,p):
    n=len(p); logit=np.log(np.maximum(p,1e-5)/np.maximum(1-p,1e-5))
    res=milp(-logit,integrality=np.ones(n,dtype=np.int8),bounds=Bounds(np.zeros(n),np.ones(n)),constraints=LinearConstraint(A,rhs,rhs),options={'time_limit':3.0,'mip_rel_gap':0.0})
    if res.x is None:return None
    return np.rint(res.x).astype(np.int8)

def main():
    j=json.load(open(ROOT/'experiments/v75_dense_block_campaign/report.json',encoding='utf-8')); p0=np.asarray(j['correctness_probabilities']); acts=j['candidate_actions']; add=np.asarray([bool(a['add_rids']) for a in acts]); ia=np.flatnonzero(add)[np.argsort(-p0[add])]; idel=np.flatnonzero(~add)[np.argsort(-p0[~add])]; idx=np.r_[ia[:71],idel[:19]]; p=np.r_[p0[ia[:71]],1-p0[idel[:19]]]; signs=np.r_[np.ones(71),-np.ones(19)]; rng=np.random.default_rng(20260821); n=len(idx); best=[]
    for density in (.25,.4,.6):
        for seed in range(1):
            A=(rng.random((28,n))<density).astype(np.int8)*signs; bits=exact=recover_pos=false_pos=skipped=0
            for t in range(10):
                y=(rng.random(n)<p).astype(np.int8); rhs=A@y; z=solve(A,rhs,p)
                if z is None: skipped+=1; continue
                bits+=int((z==y).sum()); exact+=int(np.array_equal(z,y)); recover_pos+=int((z&y).sum()); false_pos+=int((z&~y).sum())
            row={'density':density,'seed':seed,'bit_accuracy':bits/max(1,10*n),'exact_rate':exact/10,'recovered_true':recover_pos/10,'false_positive':false_pos/10,'skipped':skipped}; print(row,flush=True); best.append(row)
    (ROOT/'experiments/v90_random_code_sim_report.json').write_text(json.dumps(best,indent=2),encoding='utf-8'); print(json.dumps(best,indent=2))
if __name__=='__main__':main()
