from __future__ import annotations
import gzip,json,sys,random
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; EXP=ROOT/'experiments'; V30=EXP/'v30_meta_stack'; REC=EXP/'v25_semantic_router/cloud_dataset/semantic_records.json.gz'; BASE=EXP/'v60_combined_checkpoint/highest_verified_combined.csv'; OUT=EXP/'v89_template_relax'; OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(EXP)); sys.path.insert(0,str(EXP/'v27_closed_loop')); sys.path.insert(0,str(V30)); from build_candidate_catalog import canonical_key,prepare_order; from build_actions import site_key; from build_cross_order_probes import load_submission,write_submission
def sites(o): return frozenset(site_key(x) for x in o['alarms'])
def pattern(refs,minsup,conf):
 if len(refs)<minsup:return None
 pats=[]
 for r in refs: pats.append(tuple(sorted(Counter(canonical_key(x) for x in r['alarms'] if x['@rid'] in r['roots']).items())))
 p,n=Counter(pats).most_common(1)[0]
 return Counter(dict(p)),n/len(refs),n if n/len(refs)>=conf else None
def predict(q,refs,scores,minsup,conf):
 z=pattern(refs,minsup,conf)
 if z is None or z[2] is None:return None
 pat=z[0]; by=defaultdict(list)
 for a in q['alarms']:by[canonical_key(a)].append(a['@rid'])
 chosen=set()
 for k,c in pat.items():
  if len(by.get(k,[]))<c:return None
  chosen.update(sorted(by[k],key=lambda rid:(-scores.get((q['id'],rid),-1e30),rid))[:c])
 return chosen
def scores(split,orders):
 v=np.load(V30/f'v30_consensus_{"oof" if split=="train" else "test"}.npy'); return {(o['id'],a['rid']):float(x) for o,xs in zip(orders,[]) for a,x in []} if False else {(o['id'],a['rid']):float(x) for o in orders for a,x in zip(o['alarms'],v[sum(len(z['alarms']) for z in orders[:orders.index(o)]):sum(len(z['alarms']) for z in orders[:orders.index(o)+1])])}
def main():
 tr=[prepare_order(p,True) for p in sorted((ROOT/'train').iterdir()) if p.is_dir()]; te=[prepare_order(p,False) for p in sorted((ROOT/'test').iterdir()) if p.is_dir()]; sc_tr={}; v=np.load(V30/'v30_consensus_oof.npy'); off=0
 for o in tr:
  for a,x in zip(o['alarms'],v[off:off+len(o['alarms'])]):sc_tr[(o['id'],a['@rid'])]=float(x)
  off+=len(o['alarms'])
 v=np.load(V30/'v30_consensus_test.npy'); sc_te={}; off=0
 for o in te:
  for a,x in zip(o['alarms'],v[off:off+len(o['alarms'])]):sc_te[(o['id'],a['@rid'])]=float(x)
  off+=len(o['alarms'])
 groups=defaultdict(list)
 for o in tr:groups[o['signature']].append(o)
 base=load_submission(BASE)[1]; report={'grid':[]}
 for disjoint in (True,False):
  for minsup in (2,3,4,5):
   for conf in (.5,.67,.8,.9,1.):
    rows=[]; tp=fp=fn=0
    for q in tr:
     refs=[r for r in groups[q['signature']] if r['id']!=q['id'] and (not disjoint or sites(r).isdisjoint(sites(q)))]
     pred=predict(q,refs,sc_tr,minsup,conf)
     if pred is None:continue
     truth=set(q['roots']); tp+=len(pred&truth); fp+=len(pred-truth); fn+=len(truth-pred); rows.append(q['id'])
    f1=2*tp/max(2*tp+fp+fn,1); testrows=[]; changes=0
    for q in te:
     pred=predict(q,groups[q['signature']],sc_te,minsup,conf)
     if pred is None:continue
     cur={x['@rid'] for x in base[q['id']]}; rem=cur-pred; add=pred-cur
     if rem or add: changes+=1; testrows.append((q,pred))
    report['grid'].append({'disjoint':disjoint,'minsup':minsup,'conf':conf,'audit_orders':len(rows),'audit_f1':f1,'tp':tp,'fp':fp,'fn':fn,'test_predicted':sum(predict(q,groups[q['signature']],sc_te,minsup,conf) is not None for q in te),'test_changed':changes})
 report['best']=sorted(report['grid'],key=lambda x:(x['audit_f1'],x['audit_orders']),reverse=True)[:20]; (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(report['best'],ensure_ascii=False,indent=2))
if __name__=='__main__':main()
