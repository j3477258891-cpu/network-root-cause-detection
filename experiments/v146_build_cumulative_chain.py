"""Build a cumulative V16 swap chain on top of the decoded-positive V145 pair."""
from __future__ import annotations
import csv,gzip,hashlib,json
from pathlib import Path
ROOT=Path(r"D:\zgyidong"); EXP=ROOT/"experiments"; BASE=EXP/"v145_verified_swap_pair/v145_verified_swap_pair_from_pair015.csv"; RECORDS=EXP/"v25_semantic_router/cloud_dataset/semantic_records.json.gz"; OUT=EXP/"v146_cumulative_chain"
TARGETS=[
 ('dd29409d-298e-451b-8231-e0c53c0e420a','#-1:8b8bf8e9-9748-4726-8fd5-e9565379f047','#-1:dd1a4fe7-36f6-40d6-976a-61e7f8d57656',0.860405194239452),
 ('6600d905-9faf-4ee3-8d44-223e9bc4dfdb','#-1:f38934ea-7530-401b-b354-bde0e9a2de71','#-1:960a3406-2124-4963-ab8c-91a9b9cd77c0',0.8343106766830353),
 ('29e6bb98-ef14-4333-920b-0cd42ff58a96','#-1:95ce5b4b-39bf-4501-82ca-9aca6b9ee799','#-1:c94052a7-1334-454b-bbd2-2dc1f0ee047b',0.6200259792031415),
 ('9ca5e12f-f73f-472c-99b3-8e74c01272fe','#-1:878ee284-26b7-41c1-b7dc-47e528d71504','#-1:247d7153-ba7f-4012-86c3-2d6b9b7e9730',0.5979608960434487),
]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ids=[];out={}
 with BASE.open(encoding='utf-8-sig') as f:
  for r in csv.DictReader(f):ids.append(r['order_id']);out[r['order_id']]=json.loads(r['output'])['rootcause']
 with gzip.open(RECORDS,'rt',encoding='utf-8') as f:rec=json.load(f)['test']
 rmap={r['order_id']:r for r in rec};OUT.mkdir(parents=True,exist_ok=True);actions=[];reports=[]
 for step,(oid,rem,add,margin) in enumerate(TARGETS,1):
  cur=out[oid];assert any(x['@rid']==rem for x in cur);assert all(x['@rid']!=add for x in cur)
  a=next(a for a in rmap[oid]['alarms'] if a['rid']==add);s=a.get('source',{});out[oid]=[x for x in cur if x['@rid']!=rem]+[{'@rid':add,'title':s.get('title',''),'location':s.get('location',''),'reason':s.get('reason','')}];actions.append({'order_id':oid,'remove_rid':rem,'add_rid':add,'v16_seed_min_margin':margin})
  roots=[x['@rid'] for v in out.values() for x in v];assert len(ids)==546 and len(roots)==1035 and len(roots)==len(set(roots))
  stem=f'v146_step{step}_v145_plus_{step}_v16';p=OUT/f'{stem}.csv'
  with p.open('w',encoding='utf-8',newline='') as f:
   w=csv.DictWriter(f,fieldnames=['order_id','output']);w.writeheader()
   for k in ids:w.writerow({'order_id':k,'output':json.dumps({'rootcause':out[k]},ensure_ascii=False)})
  expected_tp=959+step;report={'version':'v146-cumulative-v16-chain','step':step,'base':str(BASE),'base_sha256':sha(BASE),'actions':list(actions),'predictions':1035,'orders':546,'conditional_expected_tp_if_all_new_swaps_positive':expected_tp,'conditional_expected_f1_if_all_new_swaps_positive':round(2*expected_tp/(1044+1035),6),'sha256':sha(p),'status':'conditional_do_not_submit_before_previous_step','warning':'V145 must first score 0.922559; each later step should only be submitted after the preceding step does not decrease.'}
  (OUT/f'{stem}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');reports.append(report)
 print(json.dumps(reports,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
