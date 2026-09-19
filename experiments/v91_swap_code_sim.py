from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
ROOT=Path(__file__).resolve().parents[1]
def solve(A,rhs,pa,pr):
 n=len(pa); ppos=pa*(1-pr); pneg=(1-pa)*pr; pzero=1-ppos-pneg
 c=np.r_[-np.log(np.maximum(pneg,1e-8)),-np.log(np.maximum(pzero,1e-8)),-np.log(np.maximum(ppos,1e-8))]
 eq=np.zeros((A.shape[0]+n,3*n)); eq[:A.shape[0],:n]=-A; eq[:A.shape[0],2*n:]=A; eq[A.shape[0]:,:n]=np.eye(n); eq[A.shape[0]:,n:2*n]=np.eye(n); eq[A.shape[0]:,2*n:]=np.eye(n)
 rr=np.r_[rhs,np.ones(n)]; res=milp(c,integrality=np.ones(3*n,dtype=np.int8),bounds=Bounds(np.zeros(3*n),np.ones(3*n)),constraints=LinearConstraint(eq,rr,rr),options={'time_limit':5.0})
 if res.x is None:
  print('milp status',res.status,res.message,flush=True)
  # print diagnostic for the caller; infeasible versus timeout matters.
  return None
 z=np.rint(res.x).reshape(3,n); return z[2]-z[0]
def main():
 j=json.load(open(ROOT/'experiments/v75_dense_block_campaign/report.json',encoding='utf-8')); p=np.asarray(j['correctness_probabilities']); acts=j['candidate_actions']; add=np.asarray([bool(a['add_rids']) for a in acts]); idx=np.flatnonzero(add)[np.argsort(-p[add])]; pa=p[idx[:40]]; rng=np.random.default_rng(20260821); n=40; out=[]
 for prmean in (.2,.3,.4,.5,.6):
  A=(rng.random((28,n))<.3).astype(np.int8); exact=acc=rec=fp=skip=0
  pr=np.full(n,prmean)
  for t in range(10):
   aa=(rng.random(n)<pa).astype(np.int8); rr=(rng.random(n)<pr).astype(np.int8); y=aa-rr; z=solve(A,A@y,pa,pr)
   if z is None:skip+=1;continue
   exact+=int(np.array_equal(z,y)); acc+=int((z==y).sum()); rec+=int((z==1).sum()*(y==1).sum() if False else ((z==1)&(y==1)).sum()); fp+=int(((z==1)&(y!=1)).sum())
  row={'pr':prmean,'exact':exact/10,'acc':acc/(10*n),'recover_pos':rec/10,'false_pos':fp/10,'skip':skip};print(row,flush=True);out.append(row)
 (ROOT/'experiments/v91_swap_code_sim_report.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
if __name__=='__main__':main()
