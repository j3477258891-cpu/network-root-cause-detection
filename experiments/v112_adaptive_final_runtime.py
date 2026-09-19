"""Generate the next of three adaptive final submissions after V104 decoding."""
import argparse,json
from pathlib import Path
import numpy as np
from v104_15day_positive_campaign import ROOT,OUT,load_actions,read_base,apply,write_csv
STATE=OUT/'adaptive_final_state.json'

def simulate(q,n=120000):
    rng=np.random.default_rng(112); return (rng.random((n,len(q)))<q).astype(np.int8)

def choose(y,rows,add_count,used):
    post=y[rows].mean(0) if len(rows) else y.mean(0); order=np.argsort(-post); best=(-1,-1,None,None)
    for k in range(30,len(post)+1,5):
        idx=np.sort(order[:k]); key=','.join(map(str,idx))
        if key in used: continue
        corr=y[rows[:,None],idx].sum(1); tp=956+add_count-(k-corr); pred=1035+add_count-k; f=2*tp/(1044+pred)
        metric=(float(np.mean(f>=.945)),float(f.mean()))
        if metric>best[:2]: best=(metric[0],metric[1],idx,key)
    return best

def main(score):
    decoded=json.load(open(OUT/'decoded.json',encoding='utf-8')); good=decoded['decoded_add_indices']; adds,pa,dels,q=load_actions(); add_count=len(good); y=simulate(q); state=json.load(open(STATE,encoding='utf-8')) if STATE.exists() else {'version':'v112-adaptive-final-1','add_count':add_count,'observations':[],'generated':[]}
    if score is not None:
        cur=state['generated'][-1]; k=len(cur['indices']); pred=1035+add_count-k; tp=round(score*(1044+pred)/2); root=956+add_count-tp; correct=k-root
        state['observations'].append({'indices':cur['indices'],'correct_nonroots':int(correct),'score':score})
        if score>=.945:
            state['achieved']=True; STATE.write_text(json.dumps(state,indent=2),encoding='utf-8'); print('target achieved',score); return
    rows=np.arange(len(y))
    for obs in state['observations']:
        idx=np.asarray(obs['indices']); rows=rows[y[rows[:,None],idx].sum(1)==obs['correct_nonroots']]
    used={','.join(map(str,x['indices'])) for x in state['generated']}; prob,mean,idx,key=choose(y,rows,add_count,used)
    if idx is None or len(state['generated'])>=3: raise RuntimeError('No adaptive final slot remains')
    number=len(state['generated'])+1; path=OUT/f'adaptive_final_{number:02d}_delete{len(idx)}.csv'; write_csv(path,apply(read_base(),[adds[i] for i in good],[dels[i] for i in idx]))
    state['generated'].append({'path':str(path),'indices':idx.tolist(),'conditional_probability':prob,'conditional_mean_f1':mean,'posterior_scenarios':int(len(rows))}); STATE.write_text(json.dumps(state,indent=2),encoding='utf-8'); print(json.dumps(state['generated'][-1],indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--score',type=float); args=ap.parse_args(); main(args.score)
