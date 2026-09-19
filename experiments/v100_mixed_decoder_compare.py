import json
from pathlib import Path
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint,linprog
ROOT=Path(__file__).resolve().parents[1]

def main():
    j75=json.load(open(ROOT/'experiments/v75_dense_block_campaign/report.json',encoding='utf-8'))
    p75=np.array(j75['correctness_probabilities']); a75=j75['candidate_actions']
    ia=np.flatnonzero([bool(a['add_rids']) for a in a75]); ia=ia[np.argsort(-p75[ia])][:71]
    j59=json.load(open(ROOT/'experiments/v59_extended_coded_campaign/report.json',encoding='utf-8'))
    p59=np.array(j59['correctness_probabilities']); a59=j59['candidate_actions']
    id=np.flatnonzero([not bool(a['add_rids']) for a in a59]); id=id[np.argsort(-(1-p59[id]))][:90]
    p=np.r_[p75[ia],1-p59[id]]; n=len(p); rng=np.random.default_rng(7)
    A=(rng.random((29,n))<.05).astype(np.int8)
    c=-np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1-1e-6))
    for t in range(5):
        y=(rng.random(n)<p).astype(int); rhs=A@y
        r=milp(c,integrality=np.ones(n,dtype=np.int8),bounds=Bounds(np.zeros(n),np.ones(n)),constraints=LinearConstraint(A,rhs,rhs),options={'time_limit':20})
        z=np.rint(r.x).astype(int)
        lp=linprog(c,A_eq=A,b_eq=rhs,bounds=(0,1),method='highs')
        rec={'acc':float((z==y).mean()),'zadd':int(z[:71].sum()),'zdel':int(z[71:].sum()),'bad_del':int(((z[71:]==1)&(y[71:]==0)).sum())}
        for th in (.5,.7,.9):
            sel=np.r_[z[:71],z[71:]*(p[71:]>=th)]; dt=int((sel[:71]*y[:71]).sum())-int((sel[71:]*(1-y[71:])).sum()); dp=int(sel[:71].sum())-int(sel[71:].sum()); rec[f'f{th}']=2*(956+dt)/(2079+dp)
        print('trial',t,'milp',rec,'lp',None if lp.x is None else {'frac':int(((lp.x>1e-3)&(lp.x<.999)).sum()),'acc05':float((np.rint(lp.x)==y).mean()),'sum':float(lp.x.sum())})

if __name__=='__main__': main()
