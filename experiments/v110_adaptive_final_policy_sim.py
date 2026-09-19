import json
from pathlib import Path
import numpy as np
from v109_adaptive_portfolio_search import candidates
ROOT=Path(__file__).resolve().parents[1]

def success(correct,k,add):
    tp=956+add-(k[None,:]-correct); pred=1035+add-k[None,:]
    return 2*tp/(1044+pred)>=.945

def main():
    meta=json.load(open(ROOT/'experiments/v104_15day_positive_campaign/campaign.json',encoding='utf-8')); q=np.asarray(meta['delete_correct_priors']); M,names=candidates(q); k=M.sum(1); rng=np.random.default_rng(110); add=33
    yt=(rng.random((60000,len(q)))<q).astype(np.int16); Rt=yt@M.T; St=success(Rt,k,add); memo={}
    def build(rows,depth):
        key=(depth,tuple(rows[:5]),len(rows))
        if depth==0 or len(rows)==0:return None
        rate=St[rows].mean(0); ci=int(np.argmax(rate)); node={'candidate':ci,'branches':{}}
        fail=rows[~St[rows,ci]]
        if depth>1:
            for obs in np.unique(Rt[fail,ci]):
                child=fail[Rt[fail,ci]==obs]
                if len(child)>=10: node['branches'][int(obs)]=build(child,depth-1)
        return node
    tree=build(np.arange(len(yt)),4)
    yv=(rng.random((100000,len(q)))<q).astype(np.int16); Rv=yv@M.T; Sv=success(Rv,k,add); ok=np.zeros(len(yv),bool)
    def walk(node,rows):
        if node is None or len(rows)==0:return
        ci=node['candidate']; ok[rows]|=Sv[rows,ci]; remain=rows[~Sv[rows,ci]]
        for obs,child in node['branches'].items(): walk(child,remain[Rv[remain,ci]==obs])
    walk(tree,np.arange(len(yv)))
    print({'add':add,'candidate_count':len(M),'adaptive_probability':float(ok.mean()),'fixed_v109_probability':json.load(open(ROOT/'experiments/v109_adaptive_portfolio.json'))['per_add_train_probability'][str(add)]})
if __name__=='__main__':main()
