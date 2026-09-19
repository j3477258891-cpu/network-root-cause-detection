"""Audit exact/relaxed order-template transfer for baseline correction (research only)."""
from __future__ import annotations
import json,re,sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(r"D:\zgyidong"); sys.path.insert(0,str(ROOT/'codexgz/work'))
import v10_grouped_ensemble as v10
TRAIN=ROOT/'train'; TEST=ROOT/'test'; OUT=ROOT/'experiments/research_pairwise_error_model'
def norm(x):
 x=str(x or ''); x=re.sub(r'[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}','<UUID>',x); return re.sub(r'\d+','#',x).strip()
def alarm_key(a,mode):
 vals=[]
 fields={'title':['title'],'title_reason':['title','reason'],'core':['title','reason','device_type','board_type','cause'],'loc':['title','reason','location'],'all':['title','reason','location','device','vendor','device_type','board_type','cause','radio','deployment','timeline']}
 for f in fields[mode]: vals.append(norm(a.get(f)))
 return tuple(vals)
def order_key(o,mode): return tuple(sorted(Counter(alarm_key(a,mode) for a in o['alarms']).items()))
def site(a):
 m=re.search(r'(?:SubNetwork|STATION|NodeMe|gNodeB)=([^,;]+)',str(a.get('location','')),re.I); return m.group(1) if m else 'unknown'
def main():
 tr=v10.load_orders(TRAIN,True); te=v10.load_orders(TEST,False)
 report={}
 for mode in ('title','title_reason','core','loc','all'):
  groups=defaultdict(list)
  for i,o in enumerate(tr):groups[order_key(o,mode)].append(i)
  test_matches=sum(1 for o in te if order_key(o,mode) in groups)
  repeated=sum(len(v)>1 for v in groups.values()); maxg=max(map(len,groups.values()))
  # leave-one-out alarm label transfer within exact template; align by key and majority.
  total=correct=covered=0; deltas=[]
  key_labels=defaultdict(list)
  for o in tr:
   for a in o['alarms']:key_labels[alarm_key(a,mode)].append(int(a.get('@rid') in o['roots']))
  for o in tr:
   for a in o['alarms']:
    vals=key_labels[alarm_key(a,mode)]
    # leave current row out; only use exact-key peers
    y=int(a.get('@rid') in o['roots'])
    if len(vals)>1:
     # Remove one occurrence approximately; this is conservative only when
     # exact duplicate signatures occur in other orders as well.
     p=(sum(vals)-y)/(len(vals)-1)
     pred=int(p>=0.5); covered+=1; correct+=pred==y; total+=1
  # candidate upper bound: test alarms whose key's train empirical p high
  high=[]; low=[]
  for oi,o in enumerate(te):
   for j,a in enumerate(o['alarms']):
    vals=key_labels.get(alarm_key(a,mode),[]); p=float(np.mean(vals)) if vals else np.nan
    (high if np.isfinite(p) and p>=.8 else low).append((p,oi,j,len(vals)))
  report[mode]={'train_groups':int(len(groups)),'repeated_groups':int(repeated),'largest_group':int(maxg),'test_exact_order_matches':int(test_matches),'alarm_peer_coverage':int(covered),'alarm_peer_acc':float(correct/max(total,1)),'test_high_p80':int(len(high)),'test_high_p90':int(sum(x[0]>=.9 for x in high)),'test_high_p80_with_support3':int(sum(x[0]>=.8 and x[3]>=3 for x in high)),'test_low_p20':int(sum(np.isfinite(x[0]) and x[0]<=.2 for x in low))}
 print(json.dumps(report,ensure_ascii=False,indent=2)); (OUT/'template_transfer_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
if __name__=='__main__':main()
