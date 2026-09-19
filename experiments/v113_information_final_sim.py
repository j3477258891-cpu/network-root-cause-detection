import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def make_masks(q):
    rng=np.random.default_rng(113); n=len(q); masks=[]; names=[]; logit=np.log(np.clip(q,1e-4,1-1e-4)/np.clip(1-q,1e-4,1-1e-4))
    for k in (70,85,100,115,130,145,160):
        for rep in range(40):
            idx=np.argsort(-(logit+rng.gumbel(size=n)*(.5+rep%3*.35)))[:k]; m=np.zeros(n,np.int8);m[idx]=1;masks.append(m);names.append((k,rep))
    return np.asarray(masks),names
def f1(add,corr,k): return 2*(956+add-(k-corr))/(1044+1035+add-k)
def main():
    meta=json.load(open(ROOT/'experiments/v104_15day_positive_campaign/campaign.json')); pa=np.array(meta['add_priors']);q=np.array(meta['delete_correct_priors']); M,names=make_masks(q); rng=np.random.default_rng(114); N=5000; add=(rng.random((N,71))<pa).sum(1); Y=(rng.random((N,len(q)))<q).astype(np.int16)
    # Focus on add_count 30-36 to avoid mixing action-count uncertainty.
    rows=np.flatnonzero((add>=30)&(add<=36)); Y=Y[rows]; add=add[rows];
    # candidate first probes; choose diverse low-overlap triples by random search.
    best=(-1,None); sets=[]
    for _ in range(50):
        ix=rng.choice(len(M),3,replace=False); inter=[]
        for j in ix:
            # exact count signatures across first probes
            inter.append(Y@M[j])
        sig=np.column_stack(inter); # group by signature; final choose posterior prefix for each group
        ok=np.zeros(len(rows),bool)
        groups={}
        for gi,key in enumerate(map(tuple,sig)): groups.setdefault(key,[]).append(gi)
        for g in groups.values():
            g=np.asarray(g); post=Y[g].mean(0); order=np.argsort(-post); bestg=np.zeros(len(g),bool)
            for k in range(40,len(q)+1,5):
                corr=Y[g]@np.eye(len(q),dtype=np.int8)[order[:k]] if False else Y[g][:,order[:k]].sum(1); s=f1(int(round(add[g].mean())),corr,k); bestg|=s>=.945
            ok[g]=bestg
        rate=ok.mean()
        if rate>best[0]:best=(rate,ix)
    print({'rows':len(rows),'best_information_policy':float(best[0]),'first_probe_names':[names[i] for i in best[1]]})
if __name__=='__main__':main()
