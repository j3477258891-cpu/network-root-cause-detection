import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def candidates(q):
    rng=np.random.default_rng(108); n=len(q); masks=[]; names=[]; order=np.argsort(-q)
    for k in range(40,n+1,5):
        m=np.zeros(n,np.int8); m[order[:k]]=1; masks.append(m); names.append(f'prefix_{k}')
    logit=np.log(np.clip(q,1e-4,1-1e-4)/np.clip(1-q,1e-4,1-1e-4))
    for k in (60,75,90,105,120,135):
        for scale in (.25,.5,1.0,1.8):
            for rep in range(25):
                idx=np.argsort(-(logit+rng.gumbel(size=n)*scale))[:k]; m=np.zeros(n,np.int8); m[idx]=1; masks.append(m); names.append(f'gumbel_k{k}_s{scale}_r{rep}')
    return np.asarray(masks,dtype=np.int8),names

def success(correct,k,add):
    tp=956+add-(k[None,:]-correct); pred=1035+add-k[None,:]
    return 2*tp/(1044+pred)>=.945

def main():
    meta=json.load(open(ROOT/'experiments/v104_15day_positive_campaign/campaign.json',encoding='utf-8')); pa=np.asarray(meta['add_priors']); q=np.asarray(meta['delete_correct_priors']); M,names=candidates(q); k=M.sum(1)
    rng=np.random.default_rng(109); y=(rng.random((50000,len(q)))<q).astype(np.int16); correct=y@M.T; mapping={}; stats={}
    for add in range(15,51):
        C=success(correct,k,add); covered=np.zeros(len(y),bool); picks=[]
        for _ in range(4):
            gain=((C&~covered[:,None]).sum(0)).astype(float); gain[picks]=-1; i=int(np.argmax(gain)); picks.append(i); covered|=C[:,i]
        mapping[str(add)]=[{'name':names[i],'delete_count':int(k[i]),'indices':np.flatnonzero(M[i]).tolist()} for i in picks]; stats[str(add)]=float(covered.mean())
    # Validate the adaptive mapping under the joint add/delete priors.
    nv=300000; av=(rng.random((nv,len(pa)))<pa).sum(1); yv=(rng.random((nv,len(q)))<q).astype(np.int16); ok=np.zeros(nv,bool)
    for add in np.unique(av):
        key=str(int(np.clip(add,15,50))); rows=np.flatnonzero(av==add); selected=mapping[key]
        for item in selected:
            idx=np.asarray(item['indices']); corr=yv[rows[:,None],idx].sum(1); kk=len(idx); tp=956+add-(kk-corr); pred=1035+add-kk; ok[rows]|=2*tp/(1044+pred)>=.945
    report={'version':'v109-adaptive-delete-portfolio-1','validation_trials':nv,'probability_reach_0_945':float(ok.mean()),'per_add_train_probability':stats,'mapping':mapping}
    (ROOT/'experiments/v109_adaptive_portfolio.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print({'probability_reach_0_945':report['probability_reach_0_945'],'candidate_count':len(M),'example_add33':[(x['name'],x['delete_count']) for x in mapping['33']]})
if __name__=='__main__':main()
