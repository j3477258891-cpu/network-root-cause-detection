import json, time
from pathlib import Path
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint

ROOT=Path(__file__).resolve().parents[1]

def solve(A, rhs, p, limit=45.0):
    # x=1 means a candidate correction is correct: add-root or delete-nonroot.
    logit=np.log(np.clip(p,1e-5,1-1e-5)/np.clip(1-p,1e-5,1-1e-5))
    res=milp(-logit, integrality=np.ones(len(p),dtype=np.int8),
             bounds=Bounds(np.zeros(len(p)),np.ones(len(p))),
             constraints=LinearConstraint(A,rhs,rhs),
             options={'time_limit':limit,'mip_rel_gap':0.0,'presolve':True})
    if res.x is None: return None, res.message
    return np.rint(res.x).astype(np.int8), res.message

def main():
    j=json.load(open(ROOT/'experiments/v75_dense_block_campaign/report.json',encoding='utf-8'))
    p0=np.asarray(j['correctness_probabilities'],float); acts=j['candidate_actions']
    isadd=np.asarray([bool(a['add_rids']) for a in acts])
    ia=np.flatnonzero(isadd)[np.argsort(-p0[isadd])]
    idel=np.flatnonzero(~isadd)[np.argsort(-(1-p0[~isadd]))]
    # retain high-information actions; delete variables are correctness probabilities 1-p0
    idx=np.r_[ia[:71], idel[:80]]
    p=np.r_[p0[ia[:71]], 1-p0[idel[:80]]]
    signs=np.r_[np.ones(71),-np.ones(80)]
    rng=np.random.default_rng(20260821)
    out=[]
    for density in (.08,.12):
      A=(rng.random((29,len(idx)))<density).astype(np.int8)*signs
      ex=acc=skip=tp=fp=0; times=[]; fs=[]; gains=[]
      for t in range(1):
        y=(rng.random(len(p))<p).astype(np.int8); rhs=A@y
        st=time.time(); z,msg=solve(A,rhs,p); times.append(time.time()-st)
        if z is None: skip+=1; continue
        ex+=int(np.array_equal(z,y)); acc+=int((z==y).sum()); tp+=int((z&y).sum()); fp+=int((z&~y).sum())
        addz=z[:71]; addy=y[:71]; delz=z[71:]; dely=y[71:]
        d_tp=int((addz&addy).sum())-int(((delz==1)&(dely==0)).sum())
        d_p=int(addz.sum())-int(delz.sum())
        f=2*(956+d_tp)/(2079+d_p); fs.append(f); gains.append((d_tp,d_p))
      row={'density':density,'n':len(p),'exact':ex/max(1,3-skip),'bit_accuracy':acc/max(1,(3-skip)*len(p)),'recover_true':tp/max(1,3-skip),'false_positive':fp/max(1,3-skip),'skip':skip,'times':times,'mean_f1':float(np.mean(fs)) if fs else None,'p945':float(np.mean(np.array(fs)>=.945)) if fs else None,'gains':gains}
      print(row,flush=True); out.append(row)
    (ROOT/'experiments/v94_mixed_code_sim.json').write_text(json.dumps(out,indent=2),encoding='utf-8')

if __name__=='__main__': main()
