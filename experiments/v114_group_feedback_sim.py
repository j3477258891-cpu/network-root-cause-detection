import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def run(groups,pa,q,seed=114):
 rng=np.random.default_rng(seed); N=150000; A=(rng.random((N,71))<pa).sum(1); Y=(rng.random((N,len(q)))<q).astype(np.int8); counts=np.column_stack([Y[:,g].sum(1) for g in groups]); subsets=[]
 for mask in range(1,8):
  idx=np.concatenate([groups[i] for i in range(3) if mask>>i&1]); subsets.append(idx)
 # train conditional policy on first 60k, validate remainder
 tr=np.arange(60000); va=np.arange(60000,N); policy={}
 for key in map(tuple,np.unique(counts[tr],axis=0)):
  rows=tr[np.all(counts[tr]==key,axis=1)]; best=(-1,0)
  for si,idx in enumerate(subsets):
   k=len(idx); corr=Y[rows[:,None],idx].sum(1); f=2*(956+A[rows]-(k-corr))/(1044+1035+A[rows]-k); rate=f.mean()
   if rate>best[0]:best=(rate,si)
  policy[key]=best[1]
 ok=[]
 for row in va:
  si=policy.get(tuple(counts[row]),0); idx=subsets[si]; k=len(idx); corr=Y[row,idx].sum(); f=2*(956+A[row]-(k-corr))/(1044+1035+A[row]-k); ok.append(f>=.945)
 return np.mean(ok)
def main():
 meta=json.load(open(ROOT/'experiments/v104_15day_positive_campaign/campaign.json')); pa=np.array(meta['add_priors']);q=np.array(meta['delete_correct_priors']); order=np.argsort(-q); n=len(q); rng=np.random.default_rng(114); best=0; bestg=None
 for t in range(5):
  perm=order.copy() if t==0 else rng.permutation(order); groups=[perm[:n//3],perm[n//3:2*n//3],perm[2*n//3:]]; p=run(groups,pa,q,200+t); 
  if p>best:best=p;bestg=[g.tolist() for g in groups]; print({'t':t,'p':p})
 print({'best':best,'groups':bestg})
if __name__=='__main__':main()
