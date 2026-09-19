import json
from pathlib import Path
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
ROOT=Path(__file__).resolve().parents[1]

def main():
    j=json.load(open(ROOT/'experiments/v75_dense_block_campaign/report.json',encoding='utf-8')); p75=np.array(j['correctness_probabilities']); a=j['candidate_actions']
    ia=np.flatnonzero([bool(x['add_rids']) for x in a]); ia=ia[np.argsort(-p75[ia])[:71]]
    j=json.load(open(ROOT/'experiments/v59_extended_coded_campaign/report.json',encoding='utf-8')); p59=np.array(j['correctness_probabilities']); a=j['candidate_actions']
    id=np.flatnonzero([not bool(x['add_rids']) for x in a]); id=id[np.argsort(-(1-p59[id]))[:90]]
    p=np.r_[p75[ia],1-p59[id]]; n=len(p); rng=np.random.default_rng(17); A=(rng.random((29,n))<.12).astype(np.int8); c=-np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1-1e-6))
    for k in [40,50,60,70]:
        fs=[]
        for t in range(10):
            y=(rng.random(n)<p).astype(int); rhs=A@y
            cons=[LinearConstraint(A,rhs,rhs),LinearConstraint(np.r_[np.zeros(71),np.ones(90)][None,:],-np.inf,k)]
            r=milp(c,integrality=np.ones(n,dtype=np.int8),bounds=Bounds(np.zeros(n),np.ones(n)),constraints=cons,options={'time_limit':20}); z=np.rint(r.x).astype(int)
            dt=int((z[:71]*y[:71]).sum())-int((z[71:]*(1-y[71:])).sum()); dp=int(z[:71].sum())-int(z[71:].sum()); fs.append(2*(956+dt)/(2079+dp))
        print(k,float(np.mean(fs)),float(np.mean(np.array(fs)>=.945)))
if __name__=='__main__':main()
