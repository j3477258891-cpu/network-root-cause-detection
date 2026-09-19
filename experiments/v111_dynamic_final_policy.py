import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def eval_success(correct,k,add):
    return 2*(956+add-(k-correct))/(1044+1035+add-k)>=.945
def main():
    meta=json.load(open(ROOT/'experiments/v104_15day_positive_campaign/campaign.json',encoding='utf-8')); q=np.asarray(meta['delete_correct_priors']); rng=np.random.default_rng(111); add=33
    yt=(rng.random((80000,len(q)))<q).astype(np.int16)
    def build(rows,depth):
        if depth==0 or len(rows)<10:return None
        post=yt[rows].mean(0); order=np.argsort(-post); best=(-1,None,None,None)
        for k in range(30,len(q)+1,5):
            mask=np.zeros(len(q),np.int16); mask[order[:k]]=1; corr=yt[rows]@mask; suc=eval_success(corr,k,add); rate=suc.mean()
            if rate>best[0]: best=(rate,mask,corr,suc)
        rate,mask,corr,suc=best; node={'indices':np.flatnonzero(mask).tolist(),'branches':{}}
        fail_rows=rows[~suc]; fail_corr=corr[~suc]
        if depth>1:
            for obs in np.unique(fail_corr):
                child=fail_rows[fail_corr==obs]
                if len(child)>=10:node['branches'][int(obs)]=build(child,depth-1)
        return node
    tree=build(np.arange(len(yt)),4); yv=(rng.random((150000,len(q)))<q).astype(np.int16); ok=np.zeros(len(yv),bool)
    def walk(node,rows):
        if node is None or not len(rows):return
        idx=np.asarray(node['indices']); k=len(idx); corr=yv[rows[:,None],idx].sum(1); suc=eval_success(corr,k,add); ok[rows]|=suc; rem=rows[~suc]; cr=corr[~suc]
        for obs,child in node['branches'].items():walk(child,rem[cr==obs])
    walk(tree,np.arange(len(yv))); out={'add':add,'adaptive_probability':float(ok.mean()),'tree':tree}; (ROOT/'experiments/v111_dynamic_final_policy_add33.json').write_text(json.dumps(out,indent=2),encoding='utf-8'); print({'add':add,'adaptive_probability':out['adaptive_probability']})
if __name__=='__main__':main()
