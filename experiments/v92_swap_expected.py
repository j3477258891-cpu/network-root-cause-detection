import gzip,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; V30=ROOT/'experiments/v30_meta_stack'; REC=ROOT/'experiments/v25_semantic_router/cloud_dataset/semantic_records.json.gz'; BASE=ROOT/'experiments/v60_combined_checkpoint/highest_verified_combined.csv'; sys.path.insert(0,str(V30)); from build_cross_order_probes import load_submission
def main():
 j=json.load(open(ROOT/'experiments/v75_dense_block_campaign/report.json',encoding='utf-8')); acts=j['candidate_actions']; pa=np.asarray(j['correctness_probabilities']); arr=np.load(ROOT/'experiments/v25_semantic_router/cloud_dataset/v25_semantic_router.npz'); tr=json.load(gzip.open(REC,'rt',encoding='utf-8'))['train']; te=json.load(gzip.open(REC,'rt',encoding='utf-8'))['test']; y=arr['train_labels'].astype(bool); so=np.load(V30/'v30_consensus_oof.npy'); st=np.load(V30/'v30_consensus_test.npy'); ptr=arr['train_alarm_ptr']; tptr=arr['test_alarm_ptr']; base=load_submission(BASE)[1]
 mask=np.zeros(len(y),bool); 
 for s,z in zip(ptr[:-1],ptr[1:]):
  s,z=int(s),int(z); q=np.argsort(-so[s:z]); mask[s+q[0]]=1
 opts=np.flatnonzero(~mask); opts=opts[np.argsort(-so[opts])]; mask[opts[:3169-int(mask.sum())]]=1
 # empirical P(root | score) among selected train rows, smoothed bins
 bins=np.quantile(so[mask],np.linspace(0,1,21)); bins=np.unique(bins); pbin=[]
 for lo,hi in zip(bins[:-1],bins[1:]+1e-8):
  q=mask&(so>=lo)&(so<hi); pbin.append((lo,hi,float(y[q].mean()) if q.any() else .5))
 def pr(x):
  for lo,hi,p in pbin:
   if lo<=x<hi:return p
  return pbin[-1][2]
 idx={o['order_id']:i for i,o in enumerate(te)}; rows=[]
 for a,p in zip(acts,pa):
  if not a['add_rids']:continue
  oi=idx[a['order_id']]; roots=base[a['order_id']]; rids={x['@rid'] for x in roots}; s,t=int(tptr[oi]),int(tptr[oi+1]); cand=[(st[s+k],q.get('@rid',q.get('rid'))) for k,q in enumerate(te[oi]['alarms']) if q.get('@rid',q.get('rid')) in rids]; rem=min(cand)[1] if cand else None; rs=min(cand)[0] if cand else 0; prr=pr(rs); rows.append({'id':a['action_id'],'pa':float(p),'remove_score':float(rs),'remove_root_prob':prr,'expected_swap_tp':float(p*(1-prr))})
 rows.sort(key=lambda x:x['expected_swap_tp'],reverse=True); print({'count':len(rows),'sum_expected_tp':sum(x['expected_swap_tp'] for x in rows),'top':rows[:10]}); (ROOT/'experiments/v92_swap_expected.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
if __name__=='__main__':main()
