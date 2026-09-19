from __future__ import annotations
import gzip,json,hashlib,sys
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; V30=ROOT/'experiments/v30_meta_stack'; V42=ROOT/'experiments/v42_marginal_count'; DATA=ROOT/'experiments/v25_semantic_router/cloud_dataset/v25_semantic_router.npz'; RECORDS=ROOT/'experiments/v25_semantic_router/cloud_dataset/semantic_records.json.gz'; OUT=ROOT/'experiments/v84_template_count_dp'; OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(V30)); sys.path.insert(0,str(V42)); from v30_meta_stack import feature_matrix; from build_cross_order_probes import load_submission,write_submission
from v42_marginal_count_allocator import ranked_context,marginal_rows,utility_matrix
MAX_ROOTS=8; BUDGETS=(1035,1044,1050)

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def forced_counts(records, train_records):
    g=defaultdict(list)
    for o in train_records:
        g[repr(o['signature'])].append(sum(int(a['is_root']) for a in o['alarms']))
    out={}; meta=[]
    for i,o in enumerate(records):
        vals=g.get(repr(o['signature']),[])
        if len(vals)<2: continue
        c=Counter(vals); k,n=c.most_common(1)[0]
        if n/len(vals)>=.9 and 1<=k<=min(MAX_ROOTS,len(o['alarms'])):
            out[i]=int(k); meta.append({'order':i,'support':len(vals),'mode':int(k),'confidence':n/len(vals)})
    return out,meta
def fixed_dp(utility,limits,forced,budget):
    n=len(limits); neg=-1e30; dp=np.full(budget+1,neg); dp[0]=0.; back=np.zeros((n,budget+1),np.int8)
    for i in range(n):
        choices=[forced[i]] if i in forced else range(1,int(limits[i])+1)
        nd=np.full(budget+1,neg); ch=np.zeros(budget+1,np.int8)
        for k in choices:
            if k>int(limits[i]): continue
            cand=dp[:-k]+utility[i,k-1]; better=cand>nd[k:]
            if np.any(better):
                rows=np.flatnonzero(better)+k; nd[rows]=cand[better]; ch[rows]=k
        dp=nd; back[i]=ch
    if dp[budget]<=neg/2: raise RuntimeError(('unreachable',budget,int(sum(forced.values()))))
    counts=np.zeros(n,np.int8); rem=budget
    for i in range(n-1,-1,-1):
        k=int(back[i,rem]); counts[i]=k; rem-=k
    return counts
def roots_for(records,ptr,score,counts):
    out={}
    for i,o in enumerate(records):
        s,t=int(ptr[i]),int(ptr[i+1]); idx=np.argsort(-score[s:t],kind='stable')[:int(counts[i])]; vals=[]
        for j in idx:
            a=o['alarms'][int(j)]; src=a.get('source',{})
            vals.append({'@rid':a['rid'],'title':src.get('title',''),'location':src.get('location',''),'reason':src.get('reason','')})
        out[o['order_id']]=vals
    return out
def main():
    with np.load(DATA) as z: arr={k:z[k] for k in z.files}
    d=json.load(gzip.open(RECORDS,'rt',encoding='utf-8')); te=d['test']; tr=d['train']; _,base=load_submission(V30/'../v60_combined_checkpoint/highest_verified_combined.csv')
    _,test_x=feature_matrix(arr,'test'); ptr=arr['test_alarm_ptr']; station=np.load(V30/'station_extra_trees_test.npy'); consensus=np.load(V30/'v30_consensus_test.npy'); domain=np.load(ROOT/'experiments/v33_domain_adaptation/station_alpha_2_test.npy'); raw=__import__('v30_meta_stack').score_matrix(arr,'test')
    sets=[station,domain,consensus]+[raw[:,i] for i in range(raw.shape[1])]; ctx=ranked_context(sets,ptr); marg=marginal_rows(test_x,station,ptr,ctx); util=utility_matrix(np.load(V42/'ensemble_test.npy'),marg,len(te)); limits=marg['limits']
    forced,meta=forced_counts(te,tr); results=[]
    for budget in BUDGETS:
        counts=fixed_dp(util,limits,forced,budget); roots=roots_for(te,ptr,consensus,counts); path=OUT/f'result_v84_template_dp_{budget}.csv'; order_ids=list(roots); write_submission(path,order_ids,roots)
        changed=sum(set(x.get('@rid') for x in roots[o]) != set(x.get('@rid') for x in base[o]) for o in roots)
        results.append({'budget':budget,'predictions':int(counts.sum()),'changed_orders':changed,'sha256':sha(path),'path':str(path),'count_distribution':{str(k):int((counts==k).sum()) for k in range(1,9)}})
    report={'version':'v84-template-count-dp-1','known_orders':len(forced),'known_distribution':dict(Counter(forced.values())),'results':results,'warning':'Uses V42 count utility with high-confidence exact-template counts; public score unverified.'}
    (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
