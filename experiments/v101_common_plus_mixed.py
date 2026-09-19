import json
from pathlib import Path
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
ROOT=Path(__file__).resolve().parents[1]
def solve(A,rhs,p):
    c=-np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1-1e-6))
    r=milp(c,integrality=np.ones(len(p),dtype=np.int8),bounds=Bounds(np.zeros(len(p)),np.ones(len(p))),constraints=LinearConstraint(A,rhs,rhs),options={'time_limit':20})
    return None if r.x is None else np.rint(r.x).astype(np.int8)
def main():
    j75=json.load(open(ROOT/'experiments/v75_dense_block_campaign/report.json',encoding='utf-8'))
    p75=np.array(j75['correctness_probabilities']); a75=j75['candidate_actions']
    ia=np.flatnonzero([bool(x['add_rids']) for x in a75]); ia=ia[np.argsort(-p75[ia])[:71]]
    j59=json.load(open(ROOT/'experiments/v59_extended_coded_campaign/report.json',encoding='utf-8'))
    p59=np.array(j59['correctness_probabilities']); a59=j59['candidate_actions']
    ids=np.flatnonzero([not bool(x['add_rids']) for x in a59]); ids=ids[np.argsort(-(1-p59[ids]))]
    common=ids[:70]; coded=ids[70:90]; p=np.r_[p75[ia],1-p59[coded]]
    rng=np.random.default_rng(22); A=(rng.random((29,len(p)))<.12).astype(np.int8); fs=[]
    for _ in range(20):
        ya=(rng.random(71)<p75[ia]).astype(int); yc=(rng.random(20)<(1-p59[coded])).astype(int); y0=(rng.random(70)<(1-p59[common])).astype(int)
        z=solve(A,A@np.r_[ya,yc],p)
        if z is None: continue
        za=z[:71]; zc=z[71:]; dt=int((za*ya).sum())-int((zc*(1-yc)).sum())-int((1-y0).sum()); dp=int(za.sum())-int(zc.sum())-70
        fs.append(2*(956+dt)/(2079+dp))
    print(float(np.mean(fs)),float(np.mean(np.array(fs)>=.945)),np.quantile(fs,[.05,.5,.95]))
if __name__=='__main__': main()
