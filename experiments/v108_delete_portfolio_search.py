import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def load():
    j=json.load(open(ROOT/'experiments/v104_15day_positive_campaign/campaign.json',encoding='utf-8'))
    return np.asarray(j['add_priors']),np.asarray(j['delete_correct_priors'])

def score_masks(pa,q,masks,n,seed):
    rng=np.random.default_rng(seed); add=(rng.random((n,len(pa)))<pa).sum(1); nonroot=(rng.random((n,len(q)))<q).astype(np.int16)
    out=[]
    for st in range(0,len(masks),64):
        M=np.asarray(masks[st:st+64],dtype=np.int16).T; k=M.sum(0); correct=nonroot@M
        tp=956+add[:,None]-(k[None,:]-correct); pred=1035+add[:,None]-k[None,:]
        out.append((2*tp/(1044+pred)>=.945))
    return np.concatenate(out,axis=1)

def main():
    pa,q=load(); rng=np.random.default_rng(108); n=len(q); masks=[]; names=[]
    order=np.argsort(-q)
    for k in range(40,n+1,5):
        m=np.zeros(n,np.int8); m[order[:k]]=1; masks.append(m); names.append(f'prefix_{k}')
    logit=np.log(np.clip(q,1e-4,1-1e-4)/np.clip(1-q,1e-4,1-1e-4))
    for k in (60,75,90,105,120,135,150,165):
        for scale in (.25,.5,1.0,1.8):
            for rep in range(25):
                s=logit+rng.gumbel(size=n)*scale; idx=np.argsort(-s)[:k]; m=np.zeros(n,np.int8); m[idx]=1
                masks.append(m); names.append(f'gumbel_k{k}_s{scale}_r{rep}')
    C=score_masks(pa,q,masks,40000,10801); covered=np.zeros(C.shape[0],bool); picks=[]
    for _ in range(4):
        gains=((C&~covered[:,None]).sum(0)).astype(float); gains[picks]=-1; i=int(np.argmax(gains)); picks.append(i); covered|=C[:,i]
    selected=[masks[i] for i in picks]; V=score_masks(pa,q,selected,300000,10802); union=V.any(1)
    report={'version':'v108-delete-portfolio-1','candidate_count':len(masks),'train_union_probability':float(covered.mean()),'validation_union_probability':float(union.mean()),'selected':[{'name':names[i],'delete_count':int(masks[i].sum()),'indices':np.flatnonzero(masks[i]).tolist()} for i in picks]}
    (ROOT/'experiments/v108_delete_portfolio.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps({k:v for k,v in report.items() if k!='selected'},indent=2)); print([(x['name'],x['delete_count']) for x in report['selected']])
if __name__=='__main__':main()
