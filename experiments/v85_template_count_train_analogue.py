from __future__ import annotations
import gzip,json,sys
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; V30=ROOT/'experiments/v30_meta_stack'; V42=ROOT/'experiments/v42_marginal_count'; DATA=ROOT/'experiments/v25_semantic_router/cloud_dataset/v25_semantic_router.npz'; RECORDS=ROOT/'experiments/v25_semantic_router/cloud_dataset/semantic_records.json.gz'; OUT=ROOT/'experiments/v85_template_count_train_analogue'; OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(V30)); sys.path.insert(0,str(V42)); from v30_meta_stack import feature_matrix,score_matrix,exact_count_mask; from v42_marginal_count_allocator import ranked_context,marginal_rows,utility_matrix
from v84_template_count_dp import fixed_dp
MAX_ROOTS=8
def force_loso(records):
 g=defaultdict(list)
 for i,o in enumerate(records): g[repr(o['signature'])].append(i)
 out={}; meta=[]
 for i,o in enumerate(records):
  vals=[sum(int(a['is_root']) for a in records[j]['alarms']) for j in g[repr(o['signature'])] if j!=i]
  if len(vals)<2: continue
  c=Counter(vals); k,n=c.most_common(1)[0]
  if n/len(vals)>=.9 and 1<=k<=min(MAX_ROOTS,len(o['alarms'])): out[i]=int(k); meta.append((i,len(vals),int(k),n/len(vals)))
 return out,meta
def select(rank,ptr,counts):
 m=np.zeros(len(rank),bool)
 for i in range(len(counts)):
  s,t=int(ptr[i]),int(ptr[i+1]); idx=np.argsort(-rank[s:t],kind='stable')[:int(counts[i])]; m[s+idx]=1
 return m
def met(m,y):
 tp=int((m&y).sum()); return {'tp':tp,'p':int(m.sum()),'truth':int(y.sum()),'f1':2*tp/(int(m.sum())+int(y.sum()))}
def main():
 with np.load(DATA) as z: arr={k:z[k] for k in z.files}
 d=json.load(gzip.open(RECORDS,'rt',encoding='utf-8')); tr=d['train']; y=arr['train_labels'].astype(bool); ptr=arr['train_alarm_ptr']; _,x=feature_matrix(arr,'train'); station=np.load(V30/'station_extra_trees_oof.npy'); consensus=np.load(V30/'v30_consensus_oof.npy'); domain=np.load(ROOT/'experiments/v33_domain_adaptation/station_alpha_2_oof.npy'); raw=score_matrix(arr,'train'); ctx=ranked_context([station,domain,consensus]+[raw[:,i] for i in range(raw.shape[1])],ptr); marg=marginal_rows(x,station,ptr,ctx); util=utility_matrix(np.load(V42/'ensemble_oof.npy'),marg,len(tr)); forced,meta=force_loso(tr); lim=marg['limits']; report={'known_orders':len(forced),'known_distribution':dict(Counter(forced.values()))}
 for budget in (3041,3169):
  m0=select(consensus,ptr,np.full(len(tr),1,np.int8)) if False else exact_count_mask(consensus,ptr,budget)
  counts=fixed_dp(util,lim,forced,budget); m=select(consensus,ptr,counts)
  report[str(budget)]={'baseline':met(m0,y),'fixed_dp':met(m,y),'delta_f1':met(m,y)['f1']-met(m0,y)['f1'],'count_accuracy':float(np.mean(counts==np.asarray([sum(a['is_root'] for a in o['alarms']) for o in tr]))),'count_mae':float(np.mean(np.abs(counts-np.asarray([sum(a['is_root'] for a in o['alarms']) for o in tr]))))}
 (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
